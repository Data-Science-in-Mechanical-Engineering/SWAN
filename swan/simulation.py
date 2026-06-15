import matplotlib.pyplot as plt
import numpy as np
import tqdm
import yaml
import pprint
import imageio.v3 as iio
from trajectory_generation import TrajectoryGenerationConfig, ACTION_DICT
from dataclasses import dataclass
import scipy
import cv2

from crazyflow import Sim
from axswarm.settings import SolverSettings
from axswarm.data import SolverData
from axswarm import solve

@dataclass
class SimulationConfig:
    trajectory_generation_config: TrajectoryGenerationConfig
    min_dist_safety_factor: float = 1.5
    eval_point_coeff: int = 8 # number of waypoints to sample per MPC horizon length
    return_dense_trajectories: bool = False # return 8Hz or 400Hz trajectories.
    solver_settings_base_path: str = "/app/static_data/axswarm/axswarm_swarm_gpt_settings.yaml" # from SwarmGPT paper
    visualization_point_radius: int = 5

def check_collisions_fast(trajectories, min_dist):
    """Checks for collisions in the trajectories
    Arguments:
        trajectories: np.array of shape (n_frames, n_drones, 3)
        min_dist: minimum distance between drones to be considered collision-free
    Returns:
        collision_drones: set_of drones that are involved in collisions at any point in time
    """
    n_frames = trajectories.shape[0]
    collision_drones = set()
    for t in tqdm.tqdm(range(n_frames), desc='Checking collisions'):
        from scipy.spatial import KDTree 
        tree = KDTree(trajectories[t])
        pairs = tree.query_pairs(r=min_dist)
        for i, j in pairs:
            collision_drones.add(i)
            collision_drones.add(j)
    return collision_drones


class CollisionlessSim(Sim):
    """
    A custom wrapper around Crazyflow's Sim that disables contact physics.
    Without this, compiling the simulation in JAX for e.g., 2000 drones on long simulations requires >200GB of RAM.
    Removal is not important as we are not interested in contact physics but just whether collisions occur, 
    which is automatically deemed a failure of our safety filter. 
    """
    def build_mjx_spec(self):
        spec = super().build_mjx_spec()
        for geom in spec.geoms:
            if geom.name != 'floor':
                geom.contype = 0
                geom.conaffinity = 0            
        return spec

def setup_waypoints_dict(trajectory_splines, t_fine):
    # Evaluate positions
    assert(t_fine[-1] > 1), 'Expected t_fine to be in seconds and not normalized to [0, 1].'
    pos = np.array([spline(t_fine) for spline in trajectory_splines])
    # Evaluate derivatives and scale by total_time using the chain rule
    vel = np.array([spline(t_fine, 1) for spline in trajectory_splines])
    acc = np.array([spline(t_fine, 2) for spline in trajectory_splines])
    # Vel and Acc should be unused because
    n_drones = pos.shape[0] # (n_drones, n_frames_fine, 3) ! different from trajectory variables elsewhere.
    t = np.tile(t_fine, (n_drones, 1))
    
    return {'time': t, 'pos': pos, 'vel': vel, 'acc': acc}

def load_axswarm_params(settings_path, print_params=False):
    with open(settings_path) as f:
        axswarm_params = yaml.safe_load(f)
    
    if print_params:
        pp = pprint.PrettyPrinter()
        print('AxSwarm parameters:')
        pp.pprint(axswarm_params)

    solver_settings = axswarm_params['SolverSettings']
    for k, v in solver_settings.items():
        if isinstance(v, list):
            solver_settings[k] = np.asarray(v)
    solver_settings = SolverSettings(**solver_settings) # SolverSettings is a dataclass
    return solver_settings, axswarm_params['Dynamics']

def run_simulation(trajectory_splines, t_frames, solver_settings, dynamics, eval_point_coeff, return_dense_trajectories, simulation_frequency=400, high_level_control_frequency=80):
    t_horizon = solver_settings.K / solver_settings.freq
    print('Time horizon:', t_horizon, 's')
    n_eval_points = int((t_frames[-1] - t_frames[0]) / t_horizon * eval_point_coeff) # eval_point_coeff = number of evaluation points per horizon length, e.g. 3

    t_waypoint_eval = np.linspace(t_frames[0], t_frames[-1], n_eval_points)
    trajectories_3d_fine = np.array([spline(t_waypoint_eval) for spline in trajectory_splines])
    trajectories_3d_fine = np.transpose(trajectories_3d_fine, (1, 0, 2)) # reshape to (n_frames_fine, n_drones, 3)
    print(f'Total show duration: {t_frames[-1] - t_frames[0]:.2f} seconds with {n_eval_points} evaluation points for MPC waypoints.')
    waypoints = setup_waypoints_dict(trajectory_splines=trajectory_splines, t_fine=t_waypoint_eval)
    assert(t_frames[-1] == waypoints['time'][0, -1])

    bounding_box_min = np.min(trajectories_3d_fine, axis=(0, 1))
    bounding_box_max = np.max(trajectories_3d_fine, axis=(0, 1))
    bounding_box_size = bounding_box_max - bounding_box_min
    # add 5% margin to bounding box size
    margin = (0.05 * bounding_box_size + 10.0) # also 10 meters absolute margin
    bounding_box_min -= margin
    bounding_box_max += margin

    # Use the .replace() method to update the frozen dataclass
    solver_settings = solver_settings.replace(
        pos_max=bounding_box_max,
        pos_min=bounding_box_min,
    )

    solver_data = SolverData.init(
        waypoints=waypoints,
        K=solver_settings.K, # steps in optimization horizon
        N=solver_settings.N, # spline order for trajectory optimization
        A=np.asarray(dynamics['A']),
        B=np.asarray(dynamics['B']),
        A_prime=np.asarray(dynamics['A_prime']),
        B_prime=np.asarray(dynamics['B_prime']),
        freq=solver_settings.freq,
        smoothness_weight=solver_settings.smoothness_weight,
        input_smoothness_weight=solver_settings.input_smoothness_weight,
        input_continuity_weight=solver_settings.input_continuity_weight,
    )

    _, n_drones, _ = trajectories_3d_fine.shape

    sim = CollisionlessSim(
        n_drones=n_drones, 
        freq=simulation_frequency, # physics frequency 
        attitude_freq=simulation_frequency, # Interal PID controller frequency
        state_freq=high_level_control_frequency, # High level control frequency, 
        control='state',
        device='cuda',
    )
    
    sim.reset() #
    control = np.zeros((sim.n_worlds, sim.n_drones, 13), dtype=np.float32)
    pos = sim.data.states.pos.at[0, ...].set(waypoints['pos'][:, 0])
    # vel = sim.data.states.vel.at[0, ...].set(waypoints['vel'][:, 0])
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos)) #, vel=vel))
    
    n_mpc_steps = int((t_frames[-1] - t_frames[0]) * solver_settings.freq)
    sub_steps = sim.freq // solver_settings.freq

    if return_dense_trajectories:
        simulated_trajectories = np.zeros((n_mpc_steps * sub_steps, n_drones, 3))
    else:
        simulated_trajectories = np.zeros((n_mpc_steps, n_drones, 3))
    success_rates = []
    success_raw = np.zeros((n_mpc_steps, n_drones))
    
    # use tqdm to show progress bar
    progress_bar = tqdm.tqdm(range(n_mpc_steps), desc='Simulating')
    for step in progress_bar:
        t = step / solver_settings.freq

        pos, vel = np.asarray(sim.data.states.pos[0]), np.asarray(sim.data.states.vel[0])
        states = np.concatenate((pos, vel), axis=-1)
        success, _, solver_data = solve(states, t, solver_data, solver_settings)
        
        if not all(success):
            success_rates.append(np.mean(success.astype(float)))
        else:
            success_rates.append(1.0)
        progress_bar.set_postfix({'MPC Success Rate': f'{success_rates[-1]*100:.1f}%'})
        success_raw[step] = success
        
        solver_data = solver_data.step(solver_data)
        control[0, :, :3] = solver_data.u_pos[:, 0]
        control[0, :, 3:6] = solver_data.u_vel[:, 0]

        sim.state_control(control)
        # you can also loop over this with sim.step(1) and export the position at each step.
    
        if return_dense_trajectories:
            for sub_step in range(sub_steps):
                sim.step(1)
                simulated_trajectories[step * sub_steps + sub_step] = sim.data.states.pos[0]
        else:
            sim.step(sub_steps)
            simulated_trajectories[step] = sim.data.states.pos[0]

    return simulated_trajectories, success_raw, success_rates, t_waypoint_eval

def transform_trajectories_back_to_image_space(trajectories, transformation_matrix):
    transformation_matrix_inv = np.linalg.inv(transformation_matrix)
    trajectories_homogeneous = np.concatenate([trajectories, np.ones_like(trajectories[..., :1])], axis=-1)
    trajectories_transformed_back = trajectories_homogeneous @ transformation_matrix_inv.T
    return trajectories_transformed_back[:, :, :3]

def rgb_to_luminosity(rgb_array):
    """
    Converts RGB to a single luminosity value using standard weights.
    This is a simple, non-quantized baseline for comparison.
    """
    if np.max(rgb_array) > 1.0:
        rgb_array = rgb_array / 255.0
    weights = np.array([0.299, 0.587, 0.114])
    luminosity = np.dot(rgb_array, weights)
    return luminosity.reshape(rgb_array.shape[:-1])

def create_colors_for_tracking_video(trajectories, visibilities, type, **kwargs):
    drone_colors = np.zeros((trajectories.shape[0], trajectories.shape[1], 3)) # (n_frames, n_drones, 3)

    if type == "first_x" or type == "first_y":
        dim_idx = 0 if type == "first_x" else 1
        cmap = plt.get_cmap('rainbow')
        val_min = np.nanmin(trajectories[:, :, dim_idx])
        val_max = np.nanmax(trajectories[:, :, dim_idx])

        for drone_idx in range(trajectories.shape[1]):
            active_frames = np.where(visibilities[:, drone_idx])[0]
            if len(active_frames) > 0:
                first_t = active_frames[0]
                first_val = trajectories[first_t, drone_idx, dim_idx]
                # Normalize the value to [0, 1] for the colormap
                norm_val = (first_val - val_min) / (val_max - val_min + 1e-6)
                drone_colors[:,drone_idx] = np.array(cmap(norm_val)[:3]) * 255
            else:
                drone_colors[:,drone_idx] = [255, 255, 255] # Fallback for completely inactive
    elif type == "constant":
        color = kwargs.get("color", (0, 255, 255)) # Default to cyan if no color provided
        drone_colors[:] = color
    elif type == "video_default":
        # sample color from the video
        video = kwargs.get("video")
        for frame_idx in range(trajectories.shape[0]):
            for drone_idx in range(trajectories.shape[1]):
                if not visibilities[frame_idx, drone_idx]:
                    continue
                x, y = int(trajectories[frame_idx, drone_idx, 0]), int(trajectories[frame_idx, drone_idx, 1])
                x = np.clip(x, 0, video.shape[2]-1)
                y = np.clip(y, 0, video.shape[1]-1)
                drone_colors[frame_idx, drone_idx] = video[frame_idx, y, x]
    else:    
        raise ValueError(f"Unknown color type: {type}")

    return drone_colors.astype(np.uint8)

def simulate_with_safety_filter(output_simulation_dir: str, input_trajectory_dir: str, simulation_config: SimulationConfig):
    min_dist_sim = simulation_config.trajectory_generation_config.min_dist_base * simulation_config.min_dist_safety_factor
    solver_settings, dynamics = load_axswarm_params(simulation_config.solver_settings_base_path)
    solver_settings = solver_settings.replace(
        vel_max=simulation_config.trajectory_generation_config.max_velocity_base,
        acc_max=simulation_config.trajectory_generation_config.max_acceleration_base,
        collision_envelope=np.full((3,), min_dist_sim)
    )

    trajectory_data = np.load(f"{input_trajectory_dir}/initial_trajectories.npz", allow_pickle=True)
    simulated_trajectories, success_raw, success_rates, t_waypoint_eval = run_simulation(
        trajectory_splines=trajectory_data['trajectory_splines'],
        t_frames=trajectory_data['t_frames'],
        solver_settings=solver_settings,
        dynamics=dynamics,
        eval_point_coeff=simulation_config.eval_point_coeff,
        return_dense_trajectories=simulation_config.return_dense_trajectories,
    )

    collision_drones = check_collisions_fast(simulated_trajectories, min_dist_sim)
    if len(collision_drones) > 0:
        print(f"Collisions detected between drones: {collision_drones}. Consider increasing the min_dist_safety_factor or adjusting the trajectory generation parameters.")
    else:
        print("No collisions detected in the simulation!")

    # save simulated trajectories and t_waypoint_eval
    np.savez(f"{output_simulation_dir}/simulated_trajectories_with_safety_filter.npz", 
        simulated_trajectories=simulated_trajectories, 
        t_waypoint_eval=t_waypoint_eval
    )

    # Save simulation_results.npz for visualization scripts
    np.savez(f"{output_simulation_dir}/simulation_results.npz",
        t_frames=trajectory_data['t_frames'],
        n_leadin_frames=trajectory_data['n_leadin_frames'],
        n_leadout_frames=trajectory_data['n_leadout_frames'],
        trajectories_simulated=simulated_trajectories,
    )

def create_video_colors(t_frames, simulated_trajectories, video, transformation_matrix, n_leadin_frames, n_leadout_frames, actions_final, keep_video_colors, action_to_color_dict=None):
    """
    t_frames_final, n_leadin_frames, n_leadout_frames and actions_final all use the same time steps
    simulated_trajectories use more time steps (~0.125) we interpolate it down. However it is still bounded by the same t_frames_final[0] and t_frames_final[-1].
    """

    simulated_trajectories_interp = np.zeros((len(t_frames), simulated_trajectories.shape[1], simulated_trajectories.shape[2]))
    for drone_idx in range(simulated_trajectories.shape[1]):
        simulated_trajectories_interp[:, drone_idx, :] = scipy.interpolate.interp1d(
            np.linspace(t_frames[0], t_frames[-1], num=simulated_trajectories.shape[0]),
            simulated_trajectories[:, drone_idx, :],
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
            kind='cubic'
        )(t_frames)

    simulated_trajectories_interp = transform_trajectories_back_to_image_space(simulated_trajectories_interp, transformation_matrix)
    simulated_trajectories_interp = simulated_trajectories_interp[:, :, 1:] # drop x

    simulated_trajectories_main_show = simulated_trajectories_interp[n_leadin_frames:-n_leadout_frames]
    print(f'Simulated trajectories shape after interpolation and trimming: {simulated_trajectories_main_show.shape}')
    # Break this down even more -> sample at the original video frame rate.
    simulated_trajectories_video_fps = scipy.interpolate.interp1d(
        t_frames[n_leadin_frames:-n_leadout_frames],
        simulated_trajectories_main_show,
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
        kind='cubic'
    )(np.linspace(t_frames[n_leadin_frames], t_frames[-n_leadout_frames], video.shape[0])) 
    print(f'Simulated trajectories shape after resampling to video FPS: {simulated_trajectories_video_fps.shape}')
    
    # Create base colors for each drone
    base_colors = create_colors_for_tracking_video(
        trajectories=simulated_trajectories_video_fps,
        visibilities=np.ones_like(simulated_trajectories_video_fps[:, :, 0], dtype=bool),
        video=video,
        type='video_default',
    )

    if not keep_video_colors:
        base_colors = rgb_to_luminosity(base_colors)

        # interpolate the base colors to fill [n_leadin_frames:n_leadout_frames]
        base_colors = scipy.interpolate.interp1d(
            np.linspace(t_frames[n_leadin_frames], t_frames[-n_leadout_frames], video.shape[0]),
            base_colors,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
            kind='nearest'
        )(t_frames[n_leadin_frames:-n_leadout_frames])

        base_colors = plt.cm.plasma(base_colors)[:, :, :3] * 255
        base_colors = base_colors.astype(np.uint8)
    else:
        base_colors = scipy.interpolate.interp1d(
            np.linspace(t_frames[n_leadin_frames], t_frames[-n_leadout_frames], video.shape[0]),
            base_colors,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
            kind='nearest'
        )(t_frames[n_leadin_frames:-n_leadout_frames])
        base_colors = base_colors.astype(np.uint8)

    colors = np.zeros((len(t_frames), simulated_trajectories.shape[1], 3), dtype=np.uint8)
    colors[n_leadin_frames:-n_leadout_frames] = base_colors
    # assign all other colors based on the actions 
    if action_to_color_dict is None:
        print("No action_to_color_dict provided, using default colors for actions.")
        action_to_color_dict = {
            ACTION_DICT['takeoff']: np.array([0, 255, 0]),  # green for takeoff
            ACTION_DICT['landing']: np.array([0, 255, 0]),  # green for land
            ACTION_DICT['transition']: np.array([255, 255, 255]),  # white for transition
            ACTION_DICT['ground']: np.array([255, 0, 0]),  # red for ground
            ACTION_DICT['undefined']: np.array([255, 0, 0])  # red for undefined
        }

    for action, color in action_to_color_dict.items():
        action_indices = np.where(actions_final == action)
        colors[action_indices] = color
    return colors

def generate_visualization_video(simulation_dir: str, trajectory_generation_dir: str, video_path: str, background_color=(0, 0, 0), point_radius=5):
    # generate the visualization video.
    video = iio.imread(video_path)
    trajectory_data = np.load(f"{trajectory_generation_dir}/initial_trajectories.npz", allow_pickle=True)
    simulation_data = np.load(f"{simulation_dir}/simulated_trajectories_with_safety_filter.npz", allow_pickle=True)

    simulated_trajectories = simulation_data['simulated_trajectories'] # (n_frames, n_drones, 3)
    t_frames = trajectory_data['t_frames'] # (n_frames,)
    # t_waypoint_eval = simulation_data['t_waypoint_eval'] # (n_eval_points,)

    simulated_trajectories_interpolated = np.zeros((len(t_frames), simulated_trajectories.shape[1], 3))
    for drone_idx in range(simulated_trajectories.shape[1]):
        simulated_trajectories_interpolated[:, drone_idx] = scipy.interpolate.interp1d(
            np.linspace(t_frames[0], t_frames[-1], num=simulated_trajectories.shape[0]),
            simulated_trajectories[:, drone_idx, :],
            axis=0)(t_frames)
    
    video_resolution = video.shape[2], video.shape[1]
    
    action_to_color_dict = {
            ACTION_DICT['takeoff']: np.array([0, 255, 0]),  # green for takeoff
            ACTION_DICT['landing']: np.array([0, 255, 0]),  # green for land
            ACTION_DICT['transition']: np.array([255, 255, 255]),  # white for transition
            ACTION_DICT['ground']: np.array([255, 0, 0]),  # red for ground
            ACTION_DICT['undefined']: np.array([255, 0, 0])  # red for undefined
    }
    
    colors = create_video_colors(
        t_frames=t_frames,
        simulated_trajectories=simulated_trajectories_interpolated,
        video=video,
        transformation_matrix=trajectory_data['transformation_matrix'],
        n_leadin_frames=trajectory_data['n_leadin_frames'],
        n_leadout_frames=trajectory_data['n_leadout_frames'],
        actions_final=trajectory_data['actions'],
        keep_video_colors=False,
        action_to_color_dict=action_to_color_dict
    )
    simulated_trajectories_interpolated_img = transform_trajectories_back_to_image_space(simulated_trajectories_interpolated, trajectory_data['transformation_matrix'])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{simulation_dir}/visualization_video.mp4", fourcc, 30.0, video_resolution)

    for frame_idx in range(simulated_trajectories_interpolated.shape[0]):
        # fill background color
        frame = np.full_like(video[0], background_color, dtype=np.uint8)
        for drone_idx in range(simulated_trajectories_interpolated_img.shape[1]):
            center = (simulated_trajectories_interpolated_img[frame_idx, drone_idx, 1].astype(int), simulated_trajectories_interpolated_img[frame_idx, drone_idx, 2].astype(int))
            # skip if the color is black (not visible) or if the point is outside the video frame
            if np.all(colors[frame_idx, drone_idx] == np.array([0, 0, 0], dtype=np.uint8)) or not (0 <= center[0] < video_resolution[0] and 0 <= center[1] < video_resolution[1]):
                continue 
            cv2.circle(frame, center, point_radius, colors[frame_idx, drone_idx].tolist(), -1)
        # if it is just a black frame, skip
        if np.all(frame == 0):
            continue
        out.write(frame)
    out.release()
    print(f"Visualization video saved to {simulation_dir}/visualization_video.mp4")
    return f"{simulation_dir}/visualization_video.mp4"
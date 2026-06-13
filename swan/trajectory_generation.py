import dataclasses
import os
import json
from dataclasses import dataclass, field, field
import tqdm
import numpy as np
import scipy
import scipy.interpolate
from scipy.interpolate import BPoly
import networkx as nx

ACTION_DICT = {
    'takeoff': 0,
    'active segment': 1,
    'transition': 2,
    'landing': 3,
    'ground': 4,
    'undefined': 5,
}
@dataclass
class SafetyFactors:
    min_dist_sample_percentile: float = 0.5
    min_dist_segment_safety_factor: float = 2.5
    min_dist_transition_safety_factor: float = 1.5
    min_dist_grid_safety_factor: float = 5
    max_velocity_segment_safety_factor: float = 0.4
    max_velocity_takeoff_landing_safety_factor: float = 0.75
    max_velocity_transition_safety_factor: float = 1.0
    max_velocity_final_safety_factor: float = 0.7
    max_acceleration_segment_safety_factor: float = 0.4
    max_acceleration_takeoff_landing_safety_factor: float = 0.75
    max_acceleration_transition_safety_factor: float = 1.0
    max_acceleration_final_safety_factor: float = 0.7
@dataclass
class BipartiteMatchingWeights:
    w_dist: float = 1.0
    w_vel_in: float = 1.0
    w_vel_out: float = 1.0

# units are in m, s, m/s, m/s^2 as appropriate.
@dataclass
class TrajectoryGenerationConfig:
    safety_factors: SafetyFactors = field(default_factory=SafetyFactors)
    bipartite_matching_weights: BipartiteMatchingWeights = field(default_factory=BipartiteMatchingWeights)
    min_dist_base: float = 0.15
    max_acceleration_base: float = 2.83
    max_velocity_base: float = 1.73
    use_urgency_cost_factor: bool = True
    sort_transitions_by_distance: bool = False
    clearance_height: float = 4.0
    output_frequency_coeff: float = 8
    segment_initial_upsample_factor: int = 10


def interpolator_factory(type, t, pos, **args):
    if type == 'bspline':
        return scipy.interpolate.make_interp_spline(t, pos, k=args['k'], bc_type=args.get('bc_type', 'clamped'))
    elif type == 'pchip':
        return scipy.interpolate.PchipInterpolator(t, pos)
    elif type == 'smoothing_spline':
        return scipy.interpolate.make_smoothing_spline(t, pos, lam=args.get('lam', None)) # lam=None, solver picks automatically 
    elif type == 'linear':
        return scipy.interpolate.interp1d(t, pos, axis=0, kind='linear')
    else:
        raise ValueError(f'Unknown interpolator type: {type}')

def get_segment_bounds(visibilities_1d, segment_starts_1d=None):
    """
    Finds the start and end indices of contiguous trajectory segments.
    Splits segments if an explicit 'segment_start' is flagged, preventing teleportation jumps.
    """
    
    padded = np.pad(visibilities_1d, (1, 1), mode='constant', constant_values=False)
    vis_starts = np.where(~padded[:-1] & padded[1:])[0]
    vis_ends = np.where(padded[:-1] & ~padded[1:])[0] - 1

    if segment_starts_1d is None or not np.any(segment_starts_1d):
        return vis_starts, vis_ends, 0

    explicit_starts = np.where(segment_starts_1d & visibilities_1d)[0]
    all_starts = np.unique(np.concatenate((vis_starts, explicit_starts)))

    all_ends = []
    for i, start_idx in enumerate(all_starts):
        # segment is forced to end right before the NEXT explicit start...
        next_start = all_starts[i+1] if i + 1 < len(all_starts) else np.inf
        # it ends naturally when visibility turns off.
        valid_vis_ends = vis_ends[vis_ends >= start_idx]
        natural_end = valid_vis_ends[0] if len(valid_vis_ends) > 0 else (len(visibilities_1d) - 1)
        # end is whichever comes first.
        actual_end = min(natural_end, next_start - 1)
        all_ends.append(actual_end)
    return all_starts, np.array(all_ends, dtype=int), len(all_starts) - len(vis_starts)

def remove_short_segments(trajectories, visibilities, segment_starts, min_segment_length):
    n_segments_removed = 0
    visibilities_cleaned = np.copy(visibilities)
    trajectories_cleaned = np.copy(trajectories)
    segment_starts_cleaned = np.copy(segment_starts)
    n_frames, n_drones = visibilities.shape

    bonus_segments = 0
    for drone_idx in range(n_drones):
        vis_1d = visibilities[:, drone_idx]
        starts_1d = segment_starts[:, drone_idx]

        starts, ends, n_new_segments = get_segment_bounds(vis_1d, starts_1d)
        bonus_segments += n_new_segments

        for start, end in zip(starts, ends):
            segment_length = end - start + 1
            if segment_length < min_segment_length:
                n_segments_removed += 1
                trajectories_cleaned[start:end+1, drone_idx] = np.nan
                visibilities_cleaned[start:end+1, drone_idx] = False
                segment_starts_cleaned[start:end+1, drone_idx] = False
    return trajectories_cleaned, visibilities_cleaned, segment_starts_cleaned, n_segments_removed

def load_image_space_trajectory_data(data_dir, min_segment_length):
    trajectories = np.load(os.path.join(data_dir, 'trajectories.npz'))['trajectories']
    visibilities = np.load(os.path.join(data_dir, 'visibilities.npz'))['visibilities']
    segment_starts = np.load(os.path.join(data_dir, 'segment_starts.npz'))['segment_starts']
    # indices
    old_indices = np.arange(trajectories.shape[1])

    always_active_indices = np.where(np.all(visibilities, axis=0))[0]
    # Reorder drones. Always active drones first, then the rest -> Simplifies bipartite matching code
    not_always_active_indices = np.setdiff1d(np.arange(trajectories.shape[1]), always_active_indices)
    new_order = np.concatenate([always_active_indices, not_always_active_indices])
    trajectories = trajectories[:, new_order, :]
    visibilities = visibilities[:, new_order]
    segment_starts = segment_starts[:, new_order]
    new_indices = old_indices[new_order]

    trajectories, visibilities, segment_starts, n_segments_removed = remove_short_segments(trajectories, visibilities, segment_starts, min_segment_length=min_segment_length)

    # trajectories here are undefined and should not be used. Use NAN to avoid silent errors
    trajectories[visibilities == 0] = np.nan

    # check for drones that are never visible and remove them
    never_visible_indices = np.where(np.sum(visibilities, axis=0) == 0)[0]
    if len(never_visible_indices) > 0:
        print(f'Warning: Found {len(never_visible_indices)} drones that are never visible. Removing them from the dataset to avoid silent errors.')
        trajectories = np.delete(trajectories, never_visible_indices, axis=1)
        visibilities = np.delete(visibilities, never_visible_indices, axis=1)
        segment_starts = np.delete(segment_starts, never_visible_indices, axis=1)
        new_indices = np.delete(new_indices, never_visible_indices)

    return trajectories, visibilities, segment_starts, len(always_active_indices), new_indices, n_segments_removed

def get_closest_distances(trajectories, activities):
    closest_distances = []
    for step in range(trajectories.shape[0]):
        active_indices = np.where(activities[step])[0]
        if len(active_indices) < 2:
            continue
        positions = trajectories[step, active_indices, :]  # shape: (n_agents, 3)
        distances = scipy.spatial.distance.pdist(positions)  # pairwise distances between agents
        for i in range(len(active_indices)):
            closest_distances.append(np.min(distances[i::len(active_indices)]))  # minimum distance to any other drone 
    return np.array(closest_distances)

def rescale_to_min_dist_improved(
    trajectories2D: np.ndarray, 
    activities: np.ndarray, 
    min_dist: float, 
    min_dist_sample_percentile: float):
    """
    Rescales trajectories to ensure a minimum distance between agents for a certain percentile of (time steps, agents).
    Args:
    - trajectory_splines (list): A list of spline functions representing the trajectories of each agent.
    - activities (np.ndarray): A boolean array of shape (n_time_steps, n_agents) indicating whether each agent is active at each time step.
    - t_eval (np.ndarray): An array of time steps at which to evaluate the splines.
    - min_dist (float): The minimum distance threshold that should be maintained between agents.
    - min_dist_sample_percentile (float): The percentile of closest distances to consider for scaling.
    """
    
    closest_distances = get_closest_distances(trajectories2D, activities)
    scaling_factor = min_dist / np.percentile(closest_distances, min_dist_sample_percentile)
    
    return scaling_factor, closest_distances

def densify_and_smooth_logical_trajectories(trajectories, visibilities, segment_starts, upsample_factor: int | None = 5, t_new=None):
    """
    Upsamples sparse logical trajectories using smoothing splines that are regularized to minimize the integral over the second derivative.
    This prevents 'spline overshoot' and reveals true paths early in the pipeline.
    
    Inputs:
    - trajectories: np.ndarray (N_frames, N_drones, 2 or 3). Contains NaNs.
    - visibilities: np.ndarray (N_frames, N_drones) boolean array.
    - segment_starts: np.ndarray (N_frames, N_drones) boolean array indicating the start of each segment.
    - upsample_factor: int. How many times denser the new array should be.
    - t_new: np.ndarray. (Optional) The time steps at which to evaluate the splines. Makes upsample_factor unused if provided.

    Outputs:
    - traj_dense: The upsampled trajectory matrix.
    - vis_dense: The upsampled visibility matrix.
    - vel_dense: The upsampled velocity matrix.
    - acc_dense: The upsampled acceleration matrix.
    """
    n_frames_old, n_drones, dims = trajectories.shape
    
    # Normalized time vectors
    t_old = np.linspace(0.0, 1.0, n_frames_old)
    if t_new is None:
        n_frames_new = ((n_frames_old - 1) * upsample_factor) + 1
        t_new = np.linspace(0.0, 1.0, n_frames_new)
    else:
        if upsample_factor is not None:
            print(f"Warning: t_new provided, upsample_factor={upsample_factor} will be ignored.")
        n_frames_new = len(t_new)
        
    # Initialize dense matrices
    traj_dense = np.full((n_frames_new, n_drones, dims), np.nan)
    vel_dense = np.full((n_frames_new, n_drones, dims), np.nan)
    acc_dense = np.full((n_frames_new, n_drones, dims), np.nan)
    vis_dense = np.zeros((n_frames_new, n_drones), dtype=bool)
    segment_starts_dense = np.zeros((n_frames_new, n_drones), dtype=bool)
    
    for drone in range(n_drones):
        # # 1. Find contiguous active segments for this drone
        # padded = np.pad(visibilities[:, drone], (1, 1), mode='constant', constant_values=False)
        # starts = np.where(~padded[:-1] & padded[1:])[0]
        # ends = np.where(padded[:-1] & ~padded[1:])[0] - 1
        starts, ends, _ = get_segment_bounds(visibilities[:, drone], segment_starts[:, drone])
        
        for start, end in zip(starts, ends):
            segment_length = end - start + 1
            
            if segment_length < 5:
                raise ValueError(f"Segment too short for smooth spline (length={segment_length}). Did you filter out short segments (<5 frames)?")
            
            t_seg = t_old[start:end+1]
            pos_seg = trajectories[start:end+1, drone, :]
            smoothing_spline = scipy.interpolate.make_smoothing_spline(t_seg, pos_seg)
            
            # find the new dense time steps that fall within this specific segment
            mask_new = (t_new >= t_seg[0] - 1e-9) & (t_new <= t_seg[-1] + 1e-9)
            t_seg_new = t_new[mask_new]
            
            # evaluate and write to the dense matrix
            traj_dense[mask_new, drone, :] = smoothing_spline(t_seg_new)
            vel_dense[mask_new, drone, :] = smoothing_spline(t_seg_new, nu=1)
            acc_dense[mask_new, drone, :] = smoothing_spline(t_seg_new, nu=2)
            vis_dense[mask_new, drone] = True
            # mark the first frame of the segment as a start
            if np.any(mask_new):
                first_new_idx = np.where(mask_new)[0][0]
                segment_starts_dense[first_new_idx, drone] = True

    return traj_dense, vis_dense, segment_starts_dense, vel_dense, acc_dense

def bipartite_matching_improved(
    trajectories_logical, velocities, accelerations, visibilities, segment_starts, 
    t_frames, max_velocity_allowed, max_acceleration_allowed, eval_freq,
    w_dist=1.0, w_vel_out=30.0, w_vel_in=30.0
):
    n_drones_logical = trajectories_logical.shape[1]
    n_extra_drones_upper_bound = n_drones_logical * 2 # very conservative estimate.

    G = build_bipartite_graph_improved(
        trajectories=trajectories_logical,
        activities=visibilities,
        segments_starts=segment_starts, 
        velocities=velocities,
        accelerations=accelerations, 
        t_frames=t_frames, 
        v_max=max_velocity_allowed,
        a_max=max_acceleration_allowed,
        n_extra_drones_upper=n_extra_drones_upper_bound,
        w_dist=w_dist, w_vel_out=w_vel_out, w_vel_in=w_vel_in,
        eval_freq=eval_freq
    )

    flow_dict = nx.min_cost_flow(G)
    unused_drones = flow_dict['source']['sink']
    n_required_extra_drones = n_extra_drones_upper_bound - unused_drones
    return flow_dict, n_required_extra_drones


def prepare_trajectories(data_dir, min_dist, min_dist_sample_percentile, vel_max_allowed, acc_max_allowed, output_frequency, clearance_height, upsample_factor, min_segment_length=5):
    assert min_segment_length >= 5, "Minimum segment length must be at least 5 frames for smooth spline fitting."

    trajectories_image_space, visibilities, segment_starts, n_always_visible, _, n_segments_removed = load_image_space_trajectory_data(data_dir, min_segment_length=min_segment_length)
    n_frames, n_logical_drones, _ = trajectories_image_space.shape

    trajectories_image_space_dense, visibilities_dense, segment_starts_dense, velocities_dense, accelerations_dense = \
        densify_and_smooth_logical_trajectories(trajectories_image_space, visibilities, segment_starts, upsample_factor=upsample_factor)
    
    space_scale_factor, closest_distances = rescale_to_min_dist_improved(
    trajectories_image_space_dense, visibilities_dense, min_dist=min_dist, min_dist_sample_percentile=min_dist_sample_percentile)

    # trajectories_image_space_dense_scaled = trajectories_image_space_dense * space_scale_factor
    velocities_dense_scaled = velocities_dense * space_scale_factor
    accelerations_dense_scaled = accelerations_dense * space_scale_factor
    max_vel_measured = np.linalg.norm(velocities_dense_scaled[visibilities_dense], axis=-1).max()
    max_acc_measured = np.linalg.norm(accelerations_dense_scaled[visibilities_dense], axis=-1).max()
    time_scaling_factor_vel = vel_max_allowed / max_vel_measured
    time_scaling_factor_acc = np.sqrt(acc_max_allowed / max_acc_measured)

    time_scaling_factor = min(time_scaling_factor_vel, time_scaling_factor_acc)
    show_duration_final = 1.0 / time_scaling_factor

    n_frames_new = np.ceil(output_frequency / time_scaling_factor).astype(int)
    
    if n_frames_new <= n_frames:
        print(f"Warning: Desired output frequency {output_frequency} Hz with time scaling factor {time_scaling_factor:.4f} results in {n_frames_new} frames. Clipping to original number of frames {n_frames}.")
    
    t_frames_final_normalized = np.linspace(0.0, 1.0, n_frames_new)

    # Use the densify function again to apply the desired sampling frequency. 
    trajectories_final, visibilities_final, segment_starts_final, velocities_final, accelerations_final = \
        densify_and_smooth_logical_trajectories(trajectories_image_space, visibilities, segment_starts, upsample_factor=None, t_new=t_frames_final_normalized)

    trajectories_final *= space_scale_factor
    velocities_final *= space_scale_factor * time_scaling_factor
    accelerations_final *= space_scale_factor * time_scaling_factor**2
    
    # move them to 3D by adding a zero x coordinate (x,y) -> (0, x,y)
    trajectories_final = np.concatenate([np.zeros_like(trajectories_final[..., :1]), trajectories_final], axis=-1)
    velocities_final = np.concatenate([np.zeros_like(velocities_final[..., :1]), velocities_final], axis=-1)
    accelerations_final = np.concatenate([np.zeros_like(accelerations_final[..., :1]), accelerations_final], axis=-1)

    # flip z
    trajectories_final[:, :, 2] *= -1
    velocities_final[:, :, 2] *= -1
    accelerations_final[:, :, 2] *= -1

    min_y = np.nanmin(trajectories_final[:, :, 1])
    max_y = np.nanmax(trajectories_final[:, :, 1])
    min_z = np.nanmin(trajectories_final[:, :, 2])
    translation_y = -(min_y + max_y) / 2 # center the trajectories around y=0
    translation_z = -min_z + clearance_height # move the lowest point to clearance_height
    trajectories_final[:, :, 1] += translation_y
    trajectories_final[:, :, 2] += translation_z

    # express the transformation as a matrix
    transformation_matrix = np.diag([space_scale_factor, space_scale_factor, -space_scale_factor, 1.0])
    transformation_matrix[1:3, 3] = [translation_y, translation_z]
        
    return trajectories_final, visibilities_final, segment_starts_final, velocities_final, accelerations_final, show_duration_final, transformation_matrix, n_always_visible, closest_distances * space_scale_factor, space_scale_factor, time_scaling_factor

def generate_quintic_transition_scipy(p0, v0, a0, p1, v1, a1, delta_t_transition, num_frames):
    """
    Generates a kinematically smooth quintic transition using SciPy.
    Returns the analytical position, velocity, and acceleration arrays.
    
    Inputs:
    - p0, v0, a0: np.ndarray (3,). Initial position, velocity, acceleration.
    - p1, v1, a1: np.ndarray (3,). Final position, velocity, acceleration.
    - delta_t_transition: float. Total time duration of the transition in seconds.
    - num_frames: int. Number of discrete frames to evaluate.
    
    Outputs:
    - pos, vel, acc: np.ndarrays of shape (num_frames, 3).
    """
    t_bounds = [0.0, delta_t_transition]
    y_bounds = [[p0, v0, a0], [p1, v1, a1]]
    
    # Construct the continuous polynomial
    quintic_poly = BPoly.from_derivatives(t_bounds, y_bounds)
    
    # Evaluate analytical derivatives
    t_frames = np.linspace(0.0, delta_t_transition, num_frames)
    pos = quintic_poly(t_frames)
    vel = quintic_poly.derivative(1)(t_frames)
    acc = quintic_poly.derivative(2)(t_frames)
    # jerk = quintic_poly.derivative(3)(t_frames)
    # snap = quintic_poly.derivative(4)(t_frames)
    # crackle = quintic_poly.derivative(5)(t_frames)

    return pos, vel, acc # , jerk, snap, crackle

def build_bipartite_graph_improved(
        trajectories, activities, segments_starts, velocities, accelerations, t_frames, 
    v_max, a_max, n_extra_drones_upper, eval_freq,
    w_dist=1.0, w_vel_out=1.0, w_vel_in=1.0
):
    n_frames, n_drones = activities.shape
    deactivations = []
    activations = []
    

    for drone in range(n_drones):
        starts, ends, _ = get_segment_bounds(activities[:, drone], segments_starts[:, drone])
        for start, end in zip(starts, ends):
            if start > 0:
                activations.append({'frame': start, 'drone': drone, 
                     'position': trajectories[start, drone], 
                     'velocity': velocities[start, drone], 
                     'acceleration': accelerations[start, drone],
                     'time': t_frames[start]
                })
            if end < n_frames - 1:
                deactivations.append({'frame': end + 1, 'drone': drone, 
                     'position': trajectories[end, drone], 
                     'velocity': velocities[end, drone], 
                     'acceleration': accelerations[end, drone],
                     'time': t_frames[end + 1]
                })
    
    transition_weights = []

    G = nx.DiGraph()
    demand_source = int(-np.sum(activities[0] == 0) - n_extra_drones_upper) # Drones available at start, i.e. all that are not already active
    demand_sink = - demand_source + len(deactivations) - len(activations) # Drones that deactivated and stay so until the end

    G.add_node('source', demand=demand_source, t=t_frames[0], layer=0) # All drones that start deactivated
    G.add_node('sink', demand=demand_sink, t=t_frames[-1] + (t_frames[1] - t_frames[0]), layer=3) # All drones that end deactivated
    G.add_edge('source', 'sink', weight=0, capacity=abs(demand_source)) # Drones that are never used
    
    for deactivation in deactivations:
        drone_idx = deactivation['drone']
        frame = deactivation['frame']
        G.add_node(f'D_{drone_idx}_{frame}', demand=-1, layer=1, **deactivation)
        G.add_edge(f'D_{drone_idx}_{frame}', 'sink', weight=0, capacity=1) # It costs nothing to keep a drone deactivated
    
    total_edge_checks = 0
    total_edge_checks_denied = 0
    total_edge_checks_denied_fast = 0
    total_edge_checks_passed = 0

    SPAWN_PENALTY = 999999999
    for activation in tqdm.tqdm(activations, total=len(activations), desc='Building bipartite graph...'):
        activation_drone_idx = activation['drone']
        activation_frame = activation['frame']
        position_in = activation['position']
        velocity_in = activation['velocity']
        acceleration_in = activation['acceleration']
        

        G.add_node(f'A_{activation_drone_idx}_{activation_frame}', demand=1, layer=2, **activation)
        G.add_edge('source', f'A_{activation_drone_idx}_{activation_frame}', weight=SPAWN_PENALTY, capacity=1) # Prefer to keep drones deactivated unless necessary

        for deactivation in deactivations:
            delta_t = activation['time'] - deactivation['time']
            if delta_t <= 0: 
                continue # Too late to react

            deactivation_drone_idx = deactivation['drone']
            deactivation_frame = deactivation['frame']
            position_out = deactivation['position']
            velocity_out = deactivation['velocity']
            acceleration_out = deactivation['acceleration']

            dist_vec = position_in - position_out
            dist = np.linalg.norm(dist_vec)
            
            total_edge_checks += 1

            t_required_lower_bound = dist / v_max
            if delta_t < t_required_lower_bound:
                total_edge_checks_denied += 1
                total_edge_checks_denied_fast += 1
                continue # rough check to avoid expensive transition time estimation
        
            _, vel, acc = generate_quintic_transition_scipy(
                p0=position_out, p1=position_in, 
                v0=velocity_out, v1=velocity_in, 
                a0=acceleration_out, a1=acceleration_in,
                delta_t_transition=delta_t, num_frames=np.ceil(delta_t * eval_freq).astype(int)
            )
            peak_vel = np.max(np.linalg.norm(vel, axis=-1))
            peak_acc = np.max(np.linalg.norm(acc, axis=-1))

            if peak_vel <= v_max and peak_acc <= a_max:
                if dist > 1e-3: # more than 1 cm
                    direction = dist_vec / dist
                    v_in_norm = np.linalg.norm(velocity_in)
                    v_out_norm = np.linalg.norm(velocity_out)
                    cos_in = np.dot(velocity_in, direction) / (1 * v_in_norm) if v_in_norm > 1e-5 else 1
                    cos_out = np.dot(velocity_out, direction) / (1 * v_out_norm) if v_out_norm > 1e-5 else 1
                    penalty_out = v_out_norm * (1 - cos_out)
                    penalty_in = v_in_norm * (1 - cos_in)
                    weight = int(dist * w_dist + penalty_out * w_vel_out + penalty_in * w_vel_in) * 1000 # Scale cost by distance and penalties (and convert to int for MCMF)
                    transition_weights.append((dist * w_dist, penalty_out * w_vel_out, penalty_in * w_vel_in))
                else:
                    weight = 0
                total_edge_checks_passed += 1
                G.add_edge(f'D_{deactivation_drone_idx}_{deactivation_frame}', f'A_{activation_drone_idx}_{activation_frame}', weight=weight, capacity=1) # Scale cost by distance (and convert to int for MCMF)
            else:
                total_edge_checks_denied += 1

    # print some stats about the edge checks
    print(f'Total potential transitions checked: {total_edge_checks}')
    print(f'Transitions denied by fast check: {total_edge_checks_denied_fast}')
    print(f'Transitions denied after full check: {total_edge_checks_denied - total_edge_checks_denied_fast}')
    print(f'Transitions allowed: {total_edge_checks_passed}')

    return G

def affine_transform_3d_points(points, transformation_matrix):
    """
    Applies an affine transformation to a set of 3D points.
    points: (n_points, 3)
    transformation_matrix: (4, 4)
    """
    homogeneous_points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    transformed_homogeneous = homogeneous_points @ transformation_matrix.T
    return transformed_homogeneous[:, :3]

def create_transition_tasks_improved(
    trajectories_logical, velocities_logical, accelerations_logical, activities_logical, segment_starts_logical, flow_dict, n_always_active_drones
):
    n_frames, n_drones_logical, _ = trajectories_logical.shape

    activation_nodes = {} # activation nodes
    start_segments = [] # segments that start at the beginning.
    
    actually_always_active_drones = []

    for logical_drone_idx in range(n_drones_logical):
        # if logical_drone_idx < n_always_active_drones:
        #     continue # handled separately

        starts, ends, _ = get_segment_bounds(activities_logical[:, logical_drone_idx], segment_starts_logical[:, logical_drone_idx])

        if (len(starts) == 1 and len(ends) == 1 and starts[0] == 0 and ends[0] == n_frames - 1):
            actually_always_active_drones.append(logical_drone_idx)
            continue # This drone is active for the entire duration

        for start, end in zip(starts, ends):
            d_node = f'D_{logical_drone_idx}_{end+1}' if end+1 < n_frames else None

            segment_info = {
                'logical_drone_idx': logical_drone_idx,
                'start_frame': start,
                'end_frame': end,
                'd_node': d_node,
            }

            if start == 0:
                start_segments.append(segment_info)
            else:
                activation_node = f'A_{logical_drone_idx}_{start}'
                activation_nodes[activation_node] = segment_info

    source_flows = flow_dict['source']
    n_extra_drones_spawned = sum(flow for target, flow in source_flows.items() if str(target).startswith('A_'))
    n_drones_physical = len(actually_always_active_drones) + len(start_segments) + n_extra_drones_spawned

    # Create matrices for physical trajectories
    actions_physical = np.zeros((n_frames, n_drones_physical), dtype=int)
    trajectories_physical = np.full((n_frames, n_drones_physical, 3), np.nan)
    velocities_physical = np.full((n_frames, n_drones_physical, 3), np.nan)
    accelerations_physical = np.full((n_frames, n_drones_physical, 3), np.nan)

    # if n_always_active_drones > 0:
    #     trajectories_physical[:, :n_always_active_drones, :] = trajectories_logical[:, :n_always_active_drones, :]
    #     velocities_physical[:, :n_always_active_drones, :] = velocities_logical[:, :n_always_active_drones, :]
    #     accelerations_physical[:, :n_always_active_drones, :] = accelerations_logical[:, :n_always_active_drones, :]
    #     actions_physical[:, :n_always_active_drones] = cfsw.ACTION_DICT['active segment']
    for idx, logical_drone_idx in enumerate(actually_always_active_drones):
        trajectories_physical[:, idx, :] = trajectories_logical[:, logical_drone_idx, :]
        velocities_physical[:, idx, :] = velocities_logical[:, logical_drone_idx, :]
        accelerations_physical[:, idx, :] = accelerations_logical[:, logical_drone_idx, :]
        actions_physical[:, idx] = ACTION_DICT['active segment']

    # All starts that do not occure at keyframe 0.
    for activation_node, flow in source_flows.items():
        if flow > 0 and str(activation_node).startswith('A_'):
            start_segments.append(activation_nodes[activation_node])

    transition_tasks = []
    # available_physical_row = n_always_active_drones
    available_physical_row = len(actually_always_active_drones)
    for current_segment in start_segments:
        physical_drone_idx = available_physical_row
        available_physical_row += 1

        while current_segment is not None:
            logical_idx = current_segment['logical_drone_idx']
            start_frame = current_segment['start_frame']
            end_frame = current_segment['end_frame']
            d_node = current_segment['d_node']

            # Copy the segment to the physical trajectory
            trajectories_physical[start_frame:end_frame+1, physical_drone_idx, :] = trajectories_logical[start_frame:end_frame+1, logical_idx, :]
            velocities_physical[start_frame:end_frame+1, physical_drone_idx, :] = velocities_logical[start_frame:end_frame+1, logical_idx, :]
            accelerations_physical[start_frame:end_frame+1, physical_drone_idx, :] = accelerations_logical[start_frame:end_frame+1, logical_idx, :]
            actions_physical[start_frame:end_frame+1, physical_drone_idx] = ACTION_DICT['active segment']

            next_segment = None

            if d_node is not None and d_node in flow_dict:
                for target_node, flow in flow_dict[d_node].items():
                    if flow > 0 and str(target_node).startswith('A_'):
                        next_segment = activation_nodes[target_node]
                        
                        # Create the transition task
                        f_next_start = next_segment['start_frame']
                        next_logical_idx = next_segment['logical_drone_idx']
                        transition_task = {
                            'f_start': end_frame,
                            'f_end': f_next_start,
                            'p_out': trajectories_logical[end_frame, logical_idx, :],
                            'v_out': velocities_logical[end_frame, logical_idx, :],
                            'a_out': accelerations_logical[end_frame, logical_idx, :],
                            'p_in': trajectories_logical[f_next_start, next_logical_idx, :],
                            'v_in': velocities_logical[f_next_start, next_logical_idx, :],
                            'a_in': accelerations_logical[f_next_start, next_logical_idx, :],
                            'physical_drone_idx': physical_drone_idx
                        }
                        transition_tasks.append(transition_task)
                        break
    
            current_segment = next_segment

    return trajectories_physical, velocities_physical, accelerations_physical, transition_tasks  

def estimate_transition_time_quintic(p0, p1, v0, v1, a0, a1, v_max, a_max, eval_freq):
    """
    Estimate the minimum time required to transition between two states in 3D using a quintic polynomial profile.
    This considers initial and final positions, velocities, and accelerations, and respects velocity and acceleration limits.
    """
    # TODO find a more efficient solution without bisection

    dist_vec = p1 - p0
    dist = np.linalg.norm(dist_vec)
    
    # edge case: the points are basically identical

    t_low = dist / v_max
    t_high = 10.0 * dist / v_max + 3.0

    max_iter = 10
    for _ in range(max_iter):
        t_mid = (t_low + t_high) / 2.0
        n_eval_frames = max(10, int(t_mid * eval_freq))
        _, vel, acc = generate_quintic_transition_scipy(
            p0=p0, p1=p1, v0=v0, v1=v1, a0=a0, a1=a1, delta_t_transition=t_mid, num_frames=n_eval_frames)
        peak_vel = np.max(np.linalg.norm(vel, axis=-1))
        peak_acc = np.max(np.linalg.norm(acc, axis=-1))

        if peak_vel > v_max or peak_acc > a_max:
            t_low = t_mid  # need more time to meet constraints
        else:
            t_high = t_mid  # can we do it faster?

    return t_high

def prepare_takeoffs_and_landings_improved(
    trajectories_physical, velocities_physical, accelerations_physical, transformation_matrix, dt_frames, v_max, a_max, global_grid_points, eval_freq, 
    use_urgency_cost_factor=False, plot_frame_distributions=False, create_random_assignments=False,
):
    """
    Prepares takeoff and landing trajectories for drones that start or end the show in an active state.
    Input:
    - trajectories_physical: np.ndarray (Total_Frames, N_Drones, 3). The global physical matrix with active segments filled in.
    - velocities_physical: np.ndarray (Total_Frames, N_Drones, 3). The global physical velocities matrix.
    - accelerations_physical: np.ndarray (Total_Frames, N_Drones, 3). The global physical accelerations matrix.
    - dt_frames: float. The time step between frames (s).
    - v_max: float. Maximum velocity for takeoff/landing (m/s).
    - a_max: float. Maximum acceleration for takeoff/landing (m/s^2).
    - transformation_matrix: np.ndarray (4, 4). Homogeneous transformation matrix to convert from world coordinates to local coordinates (local = Drone show is in the YZ-plane with Z-up and X as the depth axis).
    - use_urgency_cost_factor: bool. Whether to use the urgency cost factor in the optimization.
    """

    inv_transform_matrix = np.linalg.inv(transformation_matrix)

    n_show_frames, n_drones_physical = trajectories_physical.shape[0], trajectories_physical.shape[1]
    valid_mask = ~np.any(np.isnan(trajectories_physical), axis=2)
    first_active_frames = np.argmax(valid_mask, axis=0)
    first_active_positions = trajectories_physical[first_active_frames, np.arange(n_drones_physical)]
    last_active_frames = n_show_frames - 1 - np.argmax(np.flip(valid_mask, axis=0), axis=0)
    last_active_positions = trajectories_physical[last_active_frames, np.arange(n_drones_physical)]

    # 1. Get global up in local coordinates
    global_up = np.array([0, 0, 1, 0]) # note this is a vector, not a point
    local_up = (inv_transform_matrix @ global_up)[:3]
    local_up = local_up / np.linalg.norm(local_up)
    
    local_grid = affine_transform_3d_points(global_grid_points, inv_transform_matrix)
    grid_anchor = local_grid[0]
    h_start = np.dot(first_active_positions - grid_anchor, local_up) # grid anchor could be any point from the ground plane
    h_end = np.dot(last_active_positions - grid_anchor, local_up)
    proj_start = first_active_positions - np.outer(h_start, local_up)
    proj_end = last_active_positions - np.outer(h_end, local_up)

    if create_random_assignments:
        assigned_takeoff_pads = local_grid[np.random.choice(local_grid.shape[0], size=n_drones_physical, replace=False)]
        assigned_landing_pads = local_grid[np.random.choice(local_grid.shape[0], size=n_drones_physical, replace=False)]
    else:
        cost_start = scipy.spatial.distance.cdist(proj_start, local_grid)
        if use_urgency_cost_factor:
            urgency_start = (n_show_frames - first_active_frames) + 1
            cost_start = cost_start * urgency_start[:, np.newaxis]
        
        _, col_indices_start = scipy.optimize.linear_sum_assignment(cost_start) 
        assigned_takeoff_pads = local_grid[col_indices_start]   

        cost_end = scipy.spatial.distance.cdist(proj_end, local_grid)
        if use_urgency_cost_factor:
            urgency_end = last_active_frames + 1
            cost_end = cost_end * urgency_end[:, np.newaxis]

        _, col_indices_end = scipy.optimize.linear_sum_assignment(cost_end)
        assigned_landing_pads = local_grid[col_indices_end]

    takeoff_clearance_points = assigned_takeoff_pads + (h_start[:, np.newaxis] * local_up)
    landing_clearance_points = assigned_landing_pads + (h_end[:, np.newaxis] * local_up)
    
    # 4. Estimate max takeoff/landing times
    n_leadin_frames = 0
    n_leadout_frames = 0
    # Create takeoff / landing tasks
    takeoff_tasks = []
    landing_tasks = []
    required_frames_in_list = []
    required_frames_out_list = []

    for i in range(n_drones_physical):
        # Time for Phase 1 (Vertical ascent) + Phase 2 (3D routing)
        t_phase1_in = estimate_transition_time_quintic(
            p1=assigned_takeoff_pads[i], 
            p0=takeoff_clearance_points[i], 
            v1=np.zeros(3), 
            v0=np.zeros(3), 
            a1=np.zeros(3), 
            a0=np.zeros(3),
            eval_freq=eval_freq, 
            v_max=v_max, a_max=a_max)
        t_phase2_in = estimate_transition_time_quintic(
            p1=takeoff_clearance_points[i], 
            p0=first_active_positions[i], 
            v0=np.zeros(3), 
            v1=velocities_physical[first_active_frames[i], i],
            a0=np.zeros(3),
            a1=accelerations_physical[first_active_frames[i], i], 
            eval_freq=eval_freq, v_max=v_max, a_max=a_max)
        frames_p1_in = int(np.ceil(t_phase1_in / dt_frames)) + 1
        frames_p2_in = int(np.ceil(t_phase2_in / dt_frames)) + 1
        required_frames_in = frames_p1_in + frames_p2_in - 1
        n_leadin_frames = max(n_leadin_frames, required_frames_in)
        required_frames_in_list.append(required_frames_in)
        
        # Time for Phase 1 (3D routing) + Phase 2 (Vertical descent)
        t_phase1_out = estimate_transition_time_quintic(
            p1=last_active_positions[i], 
            p0=landing_clearance_points[i], 
            v0=velocities_physical[last_active_frames[i], i],
            v1=np.zeros(3), 
            a0=accelerations_physical[last_active_frames[i], i], 
            a1=np.zeros(3), 
            eval_freq=eval_freq, 
            v_max=v_max, a_max=a_max)
        t_phase2_out = estimate_transition_time_quintic(
            p1=landing_clearance_points[i], 
            p0=assigned_landing_pads[i], 
            v0=np.zeros(3), 
            v1=np.zeros(3), 
            a0=np.zeros(3), 
            a1=np.zeros(3), 
            eval_freq=eval_freq, 
            v_max=v_max, a_max=a_max)
        frames_p1_out = int(np.ceil(t_phase1_out / dt_frames)) + 1
        frames_p2_out = int(np.ceil(t_phase2_out / dt_frames)) + 1
        required_frames_out = frames_p1_out + frames_p2_out - 1
        n_leadout_frames = max(n_leadout_frames, required_frames_out)
        required_frames_out_list.append(required_frames_out)

        takeoff_tasks.append({
            'physical_drone_idx': i,
            'p_pad': assigned_takeoff_pads[i],
            'p_clearance': takeoff_clearance_points[i],
            'p_active': first_active_positions[i],
            'v_active': velocities_physical[first_active_frames[i], i],
            'a_active': accelerations_physical[first_active_frames[i], i],
            'f_active': first_active_frames[i],
            'f_phase1_in': frames_p1_in,
            'f_phase2_in': frames_p2_in
        })

        landing_tasks.append({
            'physical_drone_idx': i,
            'p_active': last_active_positions[i],
            'v_active': velocities_physical[last_active_frames[i], i],
            'a_active': accelerations_physical[last_active_frames[i], i],
            'p_clearance': landing_clearance_points[i],
            'p_pad': assigned_landing_pads[i],
            'f_active': last_active_frames[i],
            'f_phase1_out': frames_p1_out,
            'f_phase2_out': frames_p2_out
        })

    n_total_frames = n_leadin_frames + n_show_frames + n_leadout_frames

    return n_total_frames, n_leadin_frames, n_leadout_frames, takeoff_tasks, landing_tasks

def eval_spline_list(spline_list: list, t_eval: np.ndarray, derivative_order: int = 0) -> np.ndarray:
    """
    Evaluates a list of splines at specified time steps.
    Args:
    - spline_list (list): A list of spline functions, where each spline is a callable that takes time as input and returns a 3D position.
    - t_eval (np.ndarray): An array of time steps at which to evaluate the splines.
    Returns:
    - np.ndarray: A 3D or 2D array of shape (n_time_steps, n_splines, dim) containing the evaluated positions of each spline at each time step.
    """
    n_splines = len(spline_list)
    n_time_steps = len(t_eval)
    dim = spline_list[0](t_eval[0]).shape[0]  # assuming all splines have the same output dimension
    trajectories = np.zeros((n_time_steps, n_splines, dim))  # shape: (n_time_steps, n_splines, dim)
    for i, spline in enumerate(spline_list):
        trajectories[:, i, :] = spline(t_eval, nu=derivative_order)  # evaluate the spline at each time step
    return trajectories

def generate_piecewise_bump_scipy(d_safe, delta_t_bump, num_frames, v_max_x, a_max_x):
    """
    Generates a piecewise quintic bump: Step Out -> Hold -> Step In.
    The rise time is dynamically calculated to be as fast as kinematically allowed,
    completely decoupled from the total transition duration.
    """
    if abs(d_safe) < 1e-2: # If the required step out is very small, skip the bump 
        return np.zeros(num_frames), np.zeros(num_frames), np.zeros(num_frames)
    
    # The mathematical peaks of a rest-to-rest quintic are:
    # V_peak = 1.875 * (Distance / Time)
    # A_peak = 5.7735 * (Distance / Time^2)
    t_rise_v = 1.875 * abs(d_safe) / v_max_x
    t_rise_a = np.sqrt(5.7735 * abs(d_safe) / a_max_x)    
    t_rise = max(t_rise_v, t_rise_a)
    t_frames = np.linspace(0, delta_t_bump, num_frames)

    # 2. Build the Piecewise Boundaries
    if 2 * t_rise >= delta_t_bump:
        # Edge Case: The transition is so short we don't have time to hold.
        # Just create a single peak in the middle.
        t_bounds = [0.0, delta_t_bump / 2.0, delta_t_bump]
        y_bounds = [
            [0.0, 0.0, 0.0],
            [d_safe, 0.0, 0.0], 
            [0.0, 0.0, 0.0]
        ]
    else:
        # Standard Case: Step Out -> Hold -> Step In
        t_bounds = [0.0, t_rise, delta_t_bump - t_rise, delta_t_bump]
        y_bounds = [
            [0.0, 0.0, 0.0],       # t = 0: Start at 0
            [d_safe, 0.0, 0.0],    # t = t_rise: Reached depth, come to a stop
            [d_safe, 0.0, 0.0],    # t = end - t_rise: Stayed at depth, prepare to return
            [0.0, 0.0, 0.0]        # t = end: Returned to 0
        ]            
    # 3. Construct and evaluate the polynomial
    bump_poly = BPoly.from_derivatives(t_bounds, y_bounds)
    pos = bump_poly(t_frames)
    vel = bump_poly.derivative(1)(t_frames)
    acc = bump_poly.derivative(2)(t_frames)

    return pos, vel, acc

def check_collision_hierarchical(proposed_traj, current_trajectories, f_start, f_end, min_dist, self_drone_idx, task_idx=None, d_safe=None):
    """
    Checks if a proposed trajectory collides with any existing drones using a fast-fail hierarchy.
    
    Inputs:
    - proposed_traj: np.ndarray of shape (T, 3). The 3D coordinates of the proposed transition.
    - current_trajectories: np.ndarray of shape (Total_Frames, N_Drones, 3). The global physical matrix.
    - f_start: int. The frame index where the transition begins.
    - f_end: int. The frame index where the transition ends.
    - min_dist: float. The absolute minimum allowed distance (meters) between two drones.
    
    Outputs:
    - bool: True if a collision occurs, False if the trajectory is perfectly safe.
    """

    # 1. Check time overlap
    trajectories_sliced = current_trajectories[f_start:f_end+1, :, :]
    active_drones_mask = ~np.all(np.isnan(trajectories_sliced[:, :, 0]), axis=0) # Is the drone active at any point during the transition?
    active_drones_mask[self_drone_idx] = False # Ignore self in collision checking
    if not np.any(active_drones_mask):
        return False # This branch will super rarely be hit.

    trajectories_active = trajectories_sliced[:, active_drones_mask, :]

    # 2. Check spatial overlap
    min_proposed_traj = np.min(proposed_traj, axis=0) - min_dist
    max_proposed_traj = np.max(proposed_traj, axis=0) + min_dist
    min_trajectories = np.nanmin(trajectories_active, axis=0)
    max_trajectories = np.nanmax(trajectories_active, axis=0)

    overlap_mask = np.all((min_proposed_traj <= max_trajectories) & (max_proposed_traj >= min_trajectories), axis=1)

    if not np.any(overlap_mask):
        return False # This branch will rarely be hit.
    
    trajectories_overlap = trajectories_active[:, overlap_mask, :]

    # 3. Check point-wise distances
    distances = np.linalg.norm(proposed_traj[:, None, :] - trajectories_overlap, axis=-1)
    invalid_frames_mask = np.isnan(trajectories_overlap[:, :, 0])
    distances[invalid_frames_mask] = np.inf
        
    if np.any(distances < min_dist):
        return True # Collision detected

    return False

def create_greedy_collision_free_transitions(
    transition_tasks, trajectories_base_lt, actions_base_lt, dt_frames, min_dist, v_max, a_max, n_leadin_frames, sort_by_distance
):
    """
    Routes all transitions using YZ Quintic polynomials + X-axis Hann window bumps, 
    iteratively dodging collisions and tracking required kinematic time-stretching.
    
    Inputs:
    - transition_tasks: list of dicts. Generated by Module 1 (contains start/end frames, positions, velocities).
    - trajectories_base_lt: np.ndarray (Total_Frames, N_Drones, 3). Pre-populated with active segments.
    - actions_base_lt: np.ndarray (Total_Frames, N_Drones). Pre-populated with action labels.
    - dt_frames: float. The physical time (seconds) per frame.
    - min_dist: float. Minimum safe distance (meters) for collision avoidance.
    - v_max: float. Absolute velocity limit of the drone (m/s).
    - a_max: float. Absolute acceleration limit of the drone (m/s^2).
    - n_leadin_frames: int. Number of frames until the active segment of the show starts
    
    Outputs:
    - trajectories_base_lt: The fully dense matrix with all transitions safely routed.
    - actions_base_lt: The fully dense matrix with all transitions labeled as 'transition'.
    - global_max_stretch_factor: float. The global time multiplier required to make the show flyable.
    """

    # process transitions starting with the shortest in time
    if sort_by_distance:
        transition_tasks.sort(key=lambda task: np.linalg.norm(task['p_in'] - task['p_out']))
    else:
        transition_tasks.sort(key=lambda task: task['f_end'] - task['f_start'])

    bump_increment = min_dist / 4.5 

    for task_idx, task in enumerate(tqdm.tqdm(transition_tasks, desc='Routing transitions...')):
        f_start = task['f_start']
        f_end = task['f_end']
        n_frames_transition = f_end - f_start + 1
        n_frames_bump = n_frames_transition + 2
        delta_t_transition = (n_frames_transition) * dt_frames
        delta_t_bump = delta_t_transition + 2 * dt_frames 

        # Calculate YZ-quintic coefficients
        base_traj, base_vel, base_acc = generate_quintic_transition_scipy(
            p0=task['p_out'], v0=task['v_out'], a0=task['a_out'],
            p1=task['p_in'], v1=task['v_in'], a1=task['a_in'],
            delta_t_transition=delta_t_transition, num_frames=n_frames_transition
        )

        max_iter = 50
        d_safe = 0.0
        
        collision = True # just to silence the unbound warning
        for i in range(max_iter):
            proposed_traj = base_traj.copy()
            
            if d_safe != 0.0:
                # Add analytical position bump to X-axis
                bump_pos, bump_vel, bump_acc = generate_piecewise_bump_scipy(
                    d_safe=d_safe, delta_t_bump=delta_t_bump, num_frames=n_frames_bump,
                    v_max_x=v_max * 0.25, a_max_x=a_max * 0.25
                )
                proposed_traj[:, 0] += bump_pos[1:-1]

            collision = check_collision_hierarchical(
                proposed_traj=proposed_traj, current_trajectories=trajectories_base_lt, 
                f_start=f_start+n_leadin_frames, f_end=f_end+n_leadin_frames, min_dist=min_dist, self_drone_idx=task['physical_drone_idx'],
                task_idx=task_idx, d_safe=d_safe
            )
            
            if not collision:
                break
                
            # Sequence generator: 0, -1, +1, -2, +2, -3, +3, ... scaled by bump_incrementF
            if task_idx % 2 == 0:
                if i % 2 != 0:
                    d_safe = - ((i // 2) + 1) * bump_increment 
                else:
                    d_safe = ((i // 2)) * bump_increment
            else:
                if i % 2 == 0:
                    d_safe = - ((i // 2)) * bump_increment 
                else:
                    d_safe = ((i // 2) + 1) * bump_increment

        proposed_vel = base_vel.copy()
        proposed_acc = base_acc.copy()
        if d_safe != 0.0:
            proposed_vel[:, 0] += bump_vel[1:-1]
            proposed_acc[:, 0] += bump_acc[1:-1]
            
        # --- COMMIT TRAJECTORY ---
        physical_idx = task['physical_drone_idx']
        trajectories_base_lt[f_start+n_leadin_frames:f_end+n_leadin_frames+1, physical_idx] = proposed_traj
        actions_base_lt[f_start+n_leadin_frames:f_end+n_leadin_frames+1, physical_idx] = ACTION_DICT['transition']
        
    return trajectories_base_lt, actions_base_lt

def create_takeoffs_and_landings(
    takeoff_tasks, landing_tasks, trajectories_base, actions_base, dt_frames, v_max, a_max, min_dist, n_leadin_frames, 
    smooth_start, sort_by_distance=False,
):
    
    # sort tasks by the distance from clearance point to first / last active point
    if sort_by_distance:
        takeoff_tasks.sort(key=lambda task: np.linalg.norm(task['p_clearance'] - task['p_active']))
        landing_tasks.sort(key=lambda task: np.linalg.norm(task['p_active'] - task['p_clearance']))
    else: # sort tasks by the total transition time required
        takeoff_tasks.sort(key=lambda task: task['f_phase1_in'] + task['f_phase2_in'])
        landing_tasks.sort(key=lambda task: task['f_phase1_out'] + task['f_phase2_out'])

    for task_idx, task in enumerate(tqdm.tqdm(takeoff_tasks, desc='Generating takeoffs...')):
        p_pad = task['p_pad']
        p_clearance = task['p_clearance']
        p_active = task['p_active']
        v_active = task['v_active']        
        a_active = task['a_active']
        drone_idx = task['physical_drone_idx']
        frame_duration_phase1 = task['f_phase1_in']
        frame_duration_phase2 = task['f_phase2_in']
        if not smooth_start:
            v_active = np.zeros(3)
            a_active = np.zeros(3)

        frame_arrival = n_leadin_frames + task['f_active']
        frame_clearance = frame_arrival - frame_duration_phase2 + 1
        f_takeoff = frame_clearance - frame_duration_phase1 + 1
        assert f_takeoff >= 0, f"Not enough lead-in frames for drone {drone_idx} takeoff. Consider increasing n_leadin_frames or reducing transition times by adjusting v_max/a_max or the assigned pads."

        # pad to clearance with vertical ascent.
        pos_p1, vel_p1, acc_p1 = generate_quintic_transition_scipy(
            p0=p_pad, v0=np.zeros(3), a0=np.zeros(3), 
            p1=p_clearance, v1=np.zeros(3), a1=np.zeros(3),
            delta_t_transition=(frame_duration_phase1) * dt_frames, num_frames=frame_duration_phase1
        )
        
        # clearance to initial active position
        pos_p2, vel_p2, acc_p2 = generate_quintic_transition_scipy(
            p0=p_clearance, v0=np.zeros(3), a0=np.zeros(3), 
            p1=p_active, v1=v_active, a1=a_active,
            delta_t_transition=(frame_duration_phase2) * dt_frames, num_frames=frame_duration_phase2
        )

        trajectories_base[:f_takeoff, drone_idx] = p_pad
        trajectories_base[f_takeoff:frame_clearance, drone_idx] = pos_p1[:-1]
        trajectories_base[frame_clearance:frame_arrival+1, drone_idx] = pos_p2
        actions_base[:f_takeoff, drone_idx] = ACTION_DICT['takeoff']
        actions_base[f_takeoff:frame_clearance, drone_idx] = ACTION_DICT['takeoff']
        actions_base[frame_clearance:frame_arrival+1, drone_idx] = ACTION_DICT['takeoff']

    for task_idx, task in enumerate(tqdm.tqdm(landing_tasks, desc='Generating landings...')):
        p_active = task['p_active']
        v_active = task['v_active']
        a_active = task['a_active']
        p_clearance = task['p_clearance']
        p_pad = task['p_pad']
        drone_idx = task['physical_drone_idx']

        if not smooth_start:
            v_active = np.zeros(3)
            a_active = np.zeros(3)

        frame_duration_phase1 = task['f_phase1_out']
        frame_duration_phase2 = task['f_phase2_out']

        frame_departure = n_leadin_frames + task['f_active']
        frame_clearance = frame_departure + frame_duration_phase1 - 1
        f_landing = frame_clearance + frame_duration_phase2 - 1

        # active position to clearance
        pos_p1, vel_p1, acc_p1 = generate_quintic_transition_scipy(
            p0=p_active, v0=v_active, a0=a_active,
            p1=p_clearance, v1=np.zeros(3), a1=np.zeros(3),
            delta_t_transition=frame_duration_phase1 * dt_frames, num_frames=frame_duration_phase1
        )

        # clearance to pad
        pos_p2, vel_p2, acc_p2 = generate_quintic_transition_scipy(
            p0=p_clearance, v0=np.zeros(3), a0=np.zeros(3), 
            p1=p_pad, v1=np.zeros(3), a1=np.zeros(3),
            delta_t_transition=frame_duration_phase2 * dt_frames, num_frames=frame_duration_phase2
        )
        
        trajectories_base[frame_departure:frame_clearance, drone_idx] = pos_p1[:-1]
        trajectories_base[frame_clearance:f_landing+1, drone_idx] = pos_p2
        trajectories_base[f_landing+1:, drone_idx] = p_pad

        actions_base[frame_departure:frame_clearance, drone_idx] = ACTION_DICT['landing']
        actions_base[frame_clearance:f_landing+1, drone_idx] = ACTION_DICT['landing']
        actions_base[f_landing+1:, drone_idx] = ACTION_DICT['landing']

    return trajectories_base, actions_base


def generate_trajectories(
    input_dir: str, # base_data
    output_dir: str,
    config: TrajectoryGenerationConfig,
):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_freq = 8 # hz, as per AxSwarm default settings

    trajectories_raw, visibilities_raw, _, _, _, _ = load_image_space_trajectory_data(input_dir, min_segment_length=5)
    
    trajectories_logical, visibilities_logical, segment_starts_logical, velocities_logical, accelerations_logical, show_duration_base, transformation_matrix, n_always_visible, closest_distances, space_scale_factor, time_scaling_factor = \
    prepare_trajectories(
        data_dir=input_dir,
        min_dist=config.min_dist_base * config.safety_factors.min_dist_segment_safety_factor,
        min_dist_sample_percentile=config.safety_factors.min_dist_sample_percentile,
        vel_max_allowed = config.max_velocity_base * config.safety_factors.max_velocity_segment_safety_factor,
        acc_max_allowed = config.max_acceleration_base * config.safety_factors.max_acceleration_segment_safety_factor,
        clearance_height=config.clearance_height,
        output_frequency=base_freq * config.output_frequency_coeff,
        upsample_factor=config.segment_initial_upsample_factor,
    )
    
    t_frames_base = np.linspace(0, show_duration_base, trajectories_logical.shape[0])

    flow_dict, n_required_extra_drones = bipartite_matching_improved(
        trajectories_logical=trajectories_logical,
        visibilities=visibilities_logical,
        segment_starts=segment_starts_logical,
        velocities=velocities_logical,
        accelerations=accelerations_logical,
        t_frames=t_frames_base,
        max_acceleration_allowed=config.max_acceleration_base * config.safety_factors.max_acceleration_transition_safety_factor,
        max_velocity_allowed=config.max_velocity_base * config.safety_factors.max_velocity_transition_safety_factor,
        w_dist=config.bipartite_matching_weights.w_dist,
        w_vel_in=config.bipartite_matching_weights.w_vel_in,
        w_vel_out=config.bipartite_matching_weights.w_vel_out,
        eval_freq=base_freq * config.output_frequency_coeff,
    )
    print(f"Number of required extra drones: {n_required_extra_drones}")

    trajectories_base, velocities_base, accelerations_base, transition_tasks = create_transition_tasks_improved(
        trajectories_logical=trajectories_logical,
        velocities_logical=velocities_logical,
        accelerations_logical=accelerations_logical,
        activities_logical=visibilities_logical,
        segment_starts_logical=segment_starts_logical,
        flow_dict=flow_dict,
        n_always_active_drones=n_always_visible
    )
    print(len(transition_tasks), "transition tasks created.")
    
    grid_spacing = config.min_dist_base * config.safety_factors.min_dist_grid_safety_factor
    rows = np.ceil(np.sqrt(trajectories_base.shape[1])).astype(int)
    cols = rows
    is_even = (rows * cols) % 2 == 0
    y_coords = np.arange(cols) * grid_spacing - grid_spacing * cols / 2 + (grid_spacing / 2 if is_even else 0)
    x_coords = np.arange(rows) * grid_spacing - grid_spacing * rows / 2 + (grid_spacing / 2 if is_even else 0)
    global_grid_points = []
    for x in x_coords:
        for y in y_coords:
            global_grid_points.append((x, y, 0)) # z = 0 for the grid points
    global_grid_points = np.array(global_grid_points)

    transformation_matrix_3D_projection_plane = np.eye(4)
    n_total_frames, n_leadin_frames, n_leadout_frames, takeoff_tasks, landing_tasks = \
    prepare_takeoffs_and_landings_improved(
        trajectories_physical=trajectories_base,
        velocities_physical=velocities_base,
        accelerations_physical=accelerations_base,
        transformation_matrix=transformation_matrix_3D_projection_plane,
        dt_frames=t_frames_base[1] - t_frames_base[0],
        v_max=config.max_velocity_base * config.safety_factors.max_velocity_takeoff_landing_safety_factor,
        a_max=config.max_acceleration_base * config.safety_factors.max_acceleration_takeoff_landing_safety_factor,
        global_grid_points=global_grid_points,
        eval_freq=base_freq * config.output_frequency_coeff,
        use_urgency_cost_factor=config.use_urgency_cost_factor,
        plot_frame_distributions=False
    )

    trajectories_base_lt = np.concatenate([
        np.full((n_leadin_frames, trajectories_base.shape[1], 3), np.nan),
        trajectories_base,
        np.full((n_leadout_frames, trajectories_base.shape[1], 3), np.nan)], axis=0)

    # Create actions.
    actions_base_lt = np.full((trajectories_base_lt.shape[0], trajectories_base_lt.shape[1]), ACTION_DICT['undefined'], dtype=np.int32)
    actions_base_lt[~np.isnan(trajectories_base_lt[:, :, 0])] = ACTION_DICT['active segment']

    trajectories_base_lt, actions_base_lt = \
    create_takeoffs_and_landings(
        takeoff_tasks=takeoff_tasks,
        landing_tasks=landing_tasks,
        trajectories_base=trajectories_base_lt.copy(),
        actions_base=actions_base_lt.copy(),
        dt_frames=t_frames_base[1] - t_frames_base[0],
        min_dist= config.min_dist_base * config.safety_factors.min_dist_transition_safety_factor,
        v_max=config.max_velocity_base * config.safety_factors.max_velocity_takeoff_landing_safety_factor, 
        a_max=config.max_acceleration_base * config.safety_factors.max_acceleration_takeoff_landing_safety_factor,
        n_leadin_frames=n_leadin_frames,
        sort_by_distance=config.sort_transitions_by_distance,
        smooth_start=True
    )

    trajectories_final, actions_final = \
    create_greedy_collision_free_transitions(
        transition_tasks=transition_tasks,
        trajectories_base_lt=trajectories_base_lt.copy(),
        actions_base_lt=actions_base_lt.copy(),
        dt_frames=t_frames_base[1] - t_frames_base[0],
        min_dist= config.min_dist_base * config.safety_factors.min_dist_transition_safety_factor,
        v_max= config.max_velocity_base * config.safety_factors.max_velocity_transition_safety_factor,
        a_max= config.max_acceleration_base * config.safety_factors.max_acceleration_transition_safety_factor,
        n_leadin_frames=n_leadin_frames,
        sort_by_distance=config.sort_transitions_by_distance,
    )

    total_show_duration_preliminary = n_total_frames * (t_frames_base[1] - t_frames_base[0])
    t_frames_final_preliminary = np.linspace(0, total_show_duration_preliminary, n_total_frames)

    trajectory_splines_final_preliminary = []
    for i in range(trajectories_final.shape[1]):
        # boundary conditions = 0 velocity and acceleration at start and end points:
        bc = ([(1, [0, 0, 0]), (2, [0, 0, 0])], [(1, [0, 0, 0]), (2, [0, 0, 0])])
        trajectory_splines_final_preliminary.append(interpolator_factory('bspline', t_frames_final_preliminary, trajectories_final[:, i, :], k=5, bc_type=bc))

    t_eval = np.linspace(t_frames_final_preliminary[0], t_frames_final_preliminary[-1], int((t_frames_final_preliminary[-1] - t_frames_final_preliminary[0]) * base_freq * config.output_frequency_coeff))
    velocities_analysis = eval_spline_list(trajectory_splines_final_preliminary, t_eval, derivative_order=1)
    accelerations_analysis = eval_spline_list(trajectory_splines_final_preliminary, t_eval, derivative_order=2)
    velocity_magnitudes = np.linalg.norm(velocities_analysis, axis=2).flatten()
    acceleration_magnitudes = np.linalg.norm(accelerations_analysis, axis=2).flatten()
    max_velocity = np.nanmax(velocity_magnitudes)
    max_acceleration = np.nanmax(acceleration_magnitudes)
    velocity_scaling_factor = max_velocity / (config.max_velocity_base * config.safety_factors.max_velocity_final_safety_factor)
    acceleration_scaling_factor = max_acceleration / (config.max_acceleration_base * config.safety_factors.max_acceleration_final_safety_factor)
    # print(f'Max velocity: {max_velocity:.2f} m/s, Max acceleration: {max_acceleration:.2f} m/s^2')
    # print(f'Velocity scaling factor: {velocity_scaling_factor:.2f}, Acceleration scaling factor: {acceleration_scaling_factor:.2f}')
    # dont forget quadratic scaling for acceleration when calculating the time scaling factor:
    time_scale_factor_velocity = velocity_scaling_factor
    time_scale_factor_acceleration = np.sqrt(acceleration_scaling_factor)
    # print(f'Time scaling factor based on velocity: {time_scale_factor_velocity:.2f}, Time scaling factor based on acceleration: {time_scale_factor_acceleration:.2f}')

    total_show_duration_final = total_show_duration_preliminary * max(1, time_scale_factor_velocity, time_scale_factor_acceleration)
    t_frames_final = np.linspace(0, total_show_duration_final, n_total_frames)
    # print(f'Total show duration (preliminary): {total_show_duration_preliminary:.2f} seconds')
    print(f'Total show duration: {total_show_duration_final:.2f} seconds')

    # create the final trajectory splines with the new time frames:
    trajectory_splines_final = []
    for i in range(trajectories_final.shape[1]):
        bc = ([(1, [0, 0, 0]), (2, [0, 0, 0])],[(1, [0, 0, 0]), (2, [0, 0, 0])])
        trajectory_splines_final.append(interpolator_factory('bspline', t_frames_final, trajectories_final[:, i, :], k=5, bc_type=bc))
    
    # write everything to a file
    np.savez(f"{output_dir}/initial_trajectories.npz",
        trajectory_splines=trajectory_splines_final,
        actions=actions_final,
        t_frames=t_frames_final,
        n_required_extra_drones=n_required_extra_drones,
        transformation_matrix=transformation_matrix,
        n_leadin_frames=n_leadin_frames,
        n_leadout_frames=n_leadout_frames,
        allow_pickle=True
    )

    return trajectory_splines_final, t_frames_final, n_required_extra_drones, transformation_matrix

"""
Extract tracked drone positions, overlay them onto video frames, and save the result as a video.
Usage:
    python visualize_tracking.py \
        --simulation_data /path/to/simulation_results.npz \
        --trajectory_data /path/to/final_trajectories.npz \
        --video /path/to/input_video.mp4 \
        --output /path/to/output_video.mp4
"""

# uv run -m scripts.visualize_tracking --simulation_data /data/drone_show/whale/simulation_results.npz --trajectory_data /data/drone_show/whale/final_trajectories.npz --video /data/drone_show/videos/humpback.mp4 --output ./scripts/temp/humpback_tracking.mp4

import argparse
import os
import numpy as np
import scipy.interpolate
import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

ACTION_DICT = {
    'takeoff': 0,
    'active segment': 1,
    'transition': 2,
    'landing': 3,
    'ground': 4,
    'undefined': 5,
}


def transform_trajectories_back_to_image_space(trajectories, transformation_matrix):
    transformation_matrix_inv = np.linalg.inv(transformation_matrix)
    trajectories_homogeneous = np.concatenate(
        [trajectories, np.ones_like(trajectories[..., :1])], axis=-1
    )
    trajectories_transformed_back = trajectories_homogeneous @ transformation_matrix_inv.T
    return trajectories_transformed_back[:, :, :3]


def rgb_to_luminosity(rgb_array):
    if np.max(rgb_array) > 1.0:
        rgb_array = rgb_array / 255.0
    weights = np.array([0.299, 0.587, 0.114])
    luminosity = np.dot(rgb_array, weights)
    return luminosity.reshape(rgb_array.shape[:-1])


def create_colors_for_tracking_video(trajectories, visibilities, color_type, **kwargs):
    drone_colors = np.zeros((trajectories.shape[0], trajectories.shape[1], 3))

    if color_type == "video_default":
        video = kwargs.get("video")
        for frame_idx in range(trajectories.shape[0]):
            for drone_idx in range(trajectories.shape[1]):
                if not visibilities[frame_idx, drone_idx]:
                    continue
                x = int(np.clip(trajectories[frame_idx, drone_idx, 0], 0, video.shape[2] - 1))
                y = int(np.clip(trajectories[frame_idx, drone_idx, 1], 0, video.shape[1] - 1))
                drone_colors[frame_idx, drone_idx] = video[frame_idx, y, x]
    elif color_type == "constant":
        color = kwargs.get("color", (0, 255, 255))
        drone_colors[:] = color
    else:
        raise ValueError(f"Unknown color type: {color_type}")

    return drone_colors.astype(np.uint8)


def get_trajectory_idx(video_frame_idx, video_length, t_frames, n_leadin_frames, n_leadout_frames):
    n_show_frames = len(t_frames) - n_leadin_frames - n_leadout_frames
    return n_leadin_frames + int(round(video_frame_idx / max(video_length - 1, 1) * (n_show_frames - 1)))


def load_data(simulation_path, trajectory_path, video_path):
    simulation_data = np.load(simulation_path, allow_pickle=True)
    trajectory_data = np.load(trajectory_path, allow_pickle=True)
    video = iio.imread(video_path)
    print(f"Video shape: {video.shape}, dtype: {video.dtype}")
    return simulation_data, trajectory_data, video


def prepare_trajectories(simulation_data, trajectory_data, video, color=(255, 0, 0)):
    """
    Interpolate simulated trajectories to match video frame count and transform
    them back to image space.  Also sample per-drone colors from the video.

    Returns:
        traj_img   : (n_video_frames, n_drones, 3) – positions in image space
        colors     : (n_video_frames, n_drones, 3) – uint8 RGB colors
        n_leadin   : int
        n_leadout  : int
    """
    t_frames = simulation_data['t_frames']
    n_leadin = int(simulation_data['n_leadin_frames'])
    n_leadout = int(simulation_data['n_leadout_frames'])
    simulated_trajectories = simulation_data['trajectories_simulated']  # (T_sim, n_drones, 3)
    transformation_matrix = trajectory_data['transformation_matrix']

    n_video_frames = video.shape[0]
    n_drones = simulated_trajectories.shape[1]
    n_show_frames = len(t_frames) - n_leadin - n_leadout

    t_show = t_frames[n_leadin: len(t_frames) - n_leadout]
    t_sim_full = np.linspace(t_frames[0], t_frames[-1], simulated_trajectories.shape[0])

    # Interpolate simulated trajectories onto the show-window time steps
    traj_show = np.zeros((n_show_frames, n_drones, 3))
    for d in range(n_drones):
        traj_show[:, d, :] = scipy.interpolate.interp1d(
            t_sim_full, simulated_trajectories[:, d, :],
            axis=0, bounds_error=False, fill_value="extrapolate", kind='cubic'
        )(t_show)

    # Transform to image space and drop the leading X dimension
    traj_img_show = transform_trajectories_back_to_image_space(traj_show, transformation_matrix)
    # traj_img_show[:, :, 0] is X (ignored), 1 is col, 2 is row

    # Resample to video FPS
    t_video = np.linspace(t_show[0], t_show[-1], n_video_frames)
    traj_img_video = np.zeros((n_video_frames, n_drones, 3))
    for d in range(n_drones):
        traj_img_video[:, d, :] = scipy.interpolate.interp1d(
            t_show, traj_img_show[:, d, :],
            axis=0, bounds_error=False, fill_value="extrapolate", kind='cubic'
        )(t_video)

    # Resample actions to video FPS (nearest-neighbour — actions are categorical)
    actions_final = trajectory_data['actions']  # (len(t_frames), n_drones)
    actions_show = actions_final[n_leadin: len(t_frames) - n_leadout, :]  # (n_show_frames, n_drones)
    actions_video = scipy.interpolate.interp1d(
        t_show, actions_show,
        axis=0, bounds_error=False, fill_value="extrapolate", kind='nearest'
    )(t_video).astype(int)  # (n_video_frames, n_drones)
    active_mask = actions_video == ACTION_DICT['active segment']  # (n_video_frames, n_drones)

    # All active drones are painted with the requested color.
    colors = np.zeros((n_video_frames, n_drones, 3), dtype=np.uint8)
    colors[active_mask] = color

    return traj_img_video, colors, active_mask, n_leadin, n_leadout


def render_frame(video_frame, positions_col_row, colors, dot_size=4):
    """
    Overlay drone positions onto a single video frame and return a rendered RGB array.

    Args:
        video_frame   : (H, W, 3) uint8 numpy array
        positions_col_row : (n_drones, 2) – column (x) and row (y) in pixel coordinates
        colors        : (n_drones, 3) uint8 RGB
        dot_size      : marker size in points
    """
    h, w = video_frame.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(video_frame)
    ax.scatter(
        positions_col_row[:, 0],
        positions_col_row[:, 1],
        s=dot_size,
        c=colors / 255.0,
        linewidths=0,
        zorder=2,
    )
    ax.axis('off')

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    rendered = np.asarray(buf)[:, :, :3]
    plt.close(fig)
    return rendered


def write_tracking_video(simulation_path, trajectory_path, video_path, output_path, dot_size=4, color=(255, 0, 0)):
    simulation_data, trajectory_data, video = load_data(simulation_path, trajectory_path, video_path)

    traj_img_video, colors, active_mask, n_leadin, n_leadout = prepare_trajectories(
        simulation_data, trajectory_data, video, color=color
    )

    n_video_frames = video.shape[0]
    print(f"Rendering {n_video_frames} frames …")

    frames_out = []
    for frame_idx in range(n_video_frames):
        # Only render drones that are active in this frame
        mask = active_mask[frame_idx]  # (n_drones,)
        positions = traj_img_video[frame_idx, mask, 1:]  # (n_active, 2): col, row
        frame_colors = colors[frame_idx, mask]  # (n_active, 3)
        rendered = render_frame(video[frame_idx], positions, frame_colors, dot_size=dot_size)
        frames_out.append(rendered)
        if (frame_idx + 1) % 10 == 0:
            print(f"  {frame_idx + 1}/{n_video_frames}", end="\r")

    print(f"\nSaving to {output_path} …")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    iio.imwrite(output_path, frames_out, fps=25, codec="libx264", quality=8)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay tracked drone positions onto a video.")
    parser.add_argument("--simulation_data", required=True, help="Path to simulation_results.npz")
    parser.add_argument("--trajectory_data", required=True, help="Path to final_trajectories.npz")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output video file")
    parser.add_argument("--dot_size", type=float, default=400, help="Marker size for drone dots")   # 250
    parser.add_argument(
        "--color", type=int, nargs=3, default=[255, 0, 0], metavar=("R", "G", "B"),
        help="RGB color for active drones (default: 255 0 0)"
    )
    args = parser.parse_args()

    write_tracking_video(
        simulation_path=args.simulation_data,
        trajectory_path=args.trajectory_data,
        video_path=args.video,
        output_path=args.output,
        dot_size=args.dot_size,
        color=tuple(args.color),
    )

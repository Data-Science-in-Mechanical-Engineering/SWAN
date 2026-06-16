import os
import tempfile

import cv2
import imageio
import imageio.v3 as iio
import numpy as np
import pandas as pd
import scipy.interpolate
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg


TRAIL_FRAMES = 30  # how many past frames to draw as a fading trail

ACTION_DICT = {
    'takeoff': 0,
    'active segment': 1,
    'transition': 2,
    'landing': 3,
    'ground': 4,
    'undefined': 5,
}


def create_colors_for_tracking_video(trajectories, visibilities, color_type, **kwargs):
    """Assign per-drone colors for tracking visualizations.

    Parameters
    ----------
    trajectories : (n_frames, n_drones, D)
    visibilities : (n_frames, n_drones)
    color_type : {"first_x", "first_y", "constant", "video_default"}
    kwargs : depends on ``color_type``;
        * ``constant``: ``color`` (defaults to cyan)
        * ``video_default``: ``video`` (full input video array)

    Returns
    -------
    drone_colors : (n_frames, n_drones, 3) uint8
    """
    drone_colors = np.zeros((trajectories.shape[0], trajectories.shape[1], 3)) # (n_frames, n_drones, 3)

    if color_type == "first_x" or color_type == "first_y":
        dim_idx = 0 if color_type == "first_x" else 1
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
    elif color_type == "constant":
        color = kwargs.get("color", (0, 255, 255)) # Default to cyan if no color provided
        drone_colors[:] = color
    elif color_type == "video_default":
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
        raise ValueError(f"Unknown color type: {color_type}")

    return drone_colors.astype(np.uint8)


def create_custom_tracking_video(video, trajectories, visibilities, colors, video_resolution=None, fps=18, path=None, trace_length=0, radius=3, background_color=(0, 0, 0)):
    """
    Renders tracking points onto a video or black background using fast OpenCV drawing.
    """
    T, N, _ = trajectories.shape
    if video_resolution is not None and video is None:
        H, W = video_resolution
    elif video is not None and video_resolution is None:
        _, H, W, _ = video.shape
    else:
        raise ValueError("Must provide either video or video_resolution (if using a single color background), but not both.")

    # Setup the output file
    if path is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = temp_file.name
    else:
        output_path = path

    # render video
    writer = imageio.get_writer(output_path, fps=fps)
    for t in range(T):
        if video is None:
            frame = np.full((H, W, 3), background_color, dtype=np.uint8)
        else:
            frame = video[t].copy()

        # 1. Draw Traces (if requested)
        if trace_length > 0:
            start_trace = max(0, t - trace_length)
            for i in range(N):
                # Skip tracing if currently invisible
                if not visibilities[t, i]:
                    continue
                for s in range(start_trace, t):
                    if visibilities[s, i] and visibilities[s+1, i]:
                        pt1 = (int(trajectories[s, i, 0]), int(trajectories[s, i, 1]))
                        pt2 = (int(trajectories[s+1, i, 0]), int(trajectories[s+1, i, 1]))
                        cv2.line(frame, pt1, pt2, tuple(colors[s, i].tolist()), 1, lineType=cv2.LINE_AA)
        # 2. Draw Current Points on top
        for i in range(N):
            if visibilities[t, i]:
                x, y = int(trajectories[t, i, 0]), int(trajectories[t, i, 1])
                cv2.circle(frame, (x, y), radius, tuple(colors[t, i].tolist()), -1, lineType=cv2.LINE_AA)
        writer.append_data(frame)
    writer.close()
    return output_path


def load_and_prepare(simulation_path: str, n_anim_frames: int = 300):
    """
    Load simulation_results.npz and resample trajectories to *n_anim_frames*
    evenly-spaced frames covering the 'show' window (excluding lead-in / lead-out).

    Returns
    -------
    traj   : (n_frames, n_drones, 3)  – x, y, z positions
    t_anim : (n_frames,)              – time axis (seconds)
    """
    sim = np.load(simulation_path, allow_pickle=True)

    t_frames  = sim['t_frames']
    n_leadin  = int(sim['n_leadin_frames'])
    n_leadout = int(sim['n_leadout_frames'])
    traj_sim  = sim['trajectories_simulated']   # (T_sim, n_drones, 3)

    n_drones   = traj_sim.shape[1]
    t_sim_full = np.linspace(t_frames[0], t_frames[-1], traj_sim.shape[0])
    t_show     = t_frames[n_leadin: len(t_frames) - n_leadout]

    # Interpolate simulation trajectories onto the show window
    n_show    = len(t_show)
    traj_show = np.zeros((n_show, n_drones, 3))
    for d in range(n_drones):
        traj_show[:, d, :] = scipy.interpolate.interp1d(
            t_sim_full, traj_sim[:, d, :],
            axis=0, bounds_error=False, fill_value="extrapolate", kind='cubic'
        )(t_show)

    # Resample to animation frame count
    t_anim    = np.linspace(t_show[0], t_show[-1], n_anim_frames)
    traj_anim = np.zeros((n_anim_frames, n_drones, 3))
    for d in range(n_drones):
        traj_anim[:, d, :] = scipy.interpolate.interp1d(
            t_show, traj_show[:, d, :],
            axis=0, bounds_error=False, fill_value="extrapolate", kind='cubic'
        )(t_anim)

    return traj_anim, t_anim


def make_animation(
    traj: np.ndarray,
    t_anim: np.ndarray,
    output_path: str,
    fps: int = 25,
    dot_size: float = 20.0,
    trail_frames: int = TRAIL_FRAMES,
):
    """
    Render and save a top-down (x, y) animation of drone trajectories.

    Parameters
    ----------
    traj        : (n_frames, n_drones, 3)
    t_anim      : (n_frames,)
    output_path : destination .mp4 file
    fps         : frames per second for the output video
    dot_size    : scatter marker area (matplotlib 's' parameter)
    trail_frames: number of past frames drawn as a fading trail per drone
    """
    n_frames, n_drones, _ = traj.shape

    x_all = traj[:, :, 0]
    y_all = traj[:, :, 1]
    pad_x = (x_all.max() - x_all.min()) * 0.05 or 1.0
    pad_y = (y_all.max() - y_all.min()) * 0.05 or 1.0
    xlim  = (x_all.min() - pad_x, x_all.max() + pad_x)
    ylim  = (y_all.min() - pad_y, y_all.max() + pad_y)

    # Assign a fixed color to each drone from a colormap
    drone_colors = cm.tab20(np.linspace(0, 1, n_drones))  # (n_drones, 4) RGBA

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x  [m]', color='white')
    ax.set_ylabel('y  [m]', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    title = ax.set_title('t = 0.00 s', color='white', fontsize=11)

    # One scatter for all drones (colored by index)
    scat = ax.scatter([], [], s=dot_size, zorder=3)

    # Trail lines: one Line2D per drone
    trail_lines = [
        ax.plot([], [], lw=0.6, alpha=0.4, color=drone_colors[d])[0]
        for d in range(n_drones)
    ]

    def _init():
        scat.set_offsets(np.empty((0, 2)))
        for line in trail_lines:
            line.set_data([], [])
        return [scat, *trail_lines, title]

    def _update(frame_idx):
        xs = traj[frame_idx, :, 0]
        ys = traj[frame_idx, :, 1]

        scat.set_offsets(np.column_stack([xs, ys]))
        scat.set_color(drone_colors)

        # Trail
        start = max(0, frame_idx - trail_frames)
        for d, line in enumerate(trail_lines):
            line.set_data(
                traj[start:frame_idx + 1, d, 0],
                traj[start:frame_idx + 1, d, 1],
            )

        title.set_text(f't = {t_anim[frame_idx]:.2f} s')
        return [scat, *trail_lines, title]

    ani = animation.FuncAnimation(
        fig, _update, frames=n_frames, init_func=_init,
        blit=True, interval=1000 / fps,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    print(f"Rendering {n_frames} frames to {output_path} …")
    ani.save(output_path, writer=writer, dpi=150)
    plt.close(fig)
    print("Done.")


def export_frames(
    traj: np.ndarray,
    t_anim: np.ndarray,
    export_dir: str,
    ratio: float,
    dot_size: float = 20.0,
    trail_frames: int = TRAIL_FRAMES,
):
    """
    Export selected frames as CSV files and corresponding scatterplot PNGs.

    The frames saved are those at normalised positions 0.0, ratio, 2*ratio, …
    up to (and including, if it lands exactly) 1.0.

    Each CSV has columns: drone_id, x, y, z.
    Each PNG shows the top-down (x, y) scatter with a trail at that instant.

    Parameters
    ----------
    traj       : (n_frames, n_drones, 3)
    t_anim     : (n_frames,)
    export_dir : directory to write files into
    ratio      : step between exported frames as a fraction of total duration
    dot_size   : scatter marker area
    trail_frames: trail length in frames
    """
    n_frames, n_drones, _ = traj.shape
    os.makedirs(export_dir, exist_ok=True)

    # Determine which frame indices to export
    n_steps   = int(round(1.0 / ratio))          # e.g. ratio=0.3 → 0.0,0.3,0.6,0.9 → 4 steps
    positions = np.arange(n_steps) * ratio       # [0.0, 0.3, 0.6, 0.9]
    frame_indices = [int(round(p * (n_frames - 1))) for p in positions]
    frame_indices = sorted(set(frame_indices))   # deduplicate / sort

    # Shared axis limits (consistent across all exported plots)
    x_all = traj[:, :, 0]
    y_all = traj[:, :, 1]
    pad_x = (x_all.max() - x_all.min()) * 0.05 or 1.0
    pad_y = (y_all.max() - y_all.min()) * 0.05 or 1.0
    xlim  = (x_all.min() - pad_x, x_all.max() + pad_x)
    ylim  = (y_all.min() - pad_y, y_all.max() + pad_y)

    drone_colors = cm.tab20(np.linspace(0, 1, n_drones))

    print(f"Exporting {len(frame_indices)} frames to {export_dir} …")
    for frame_idx in frame_indices:
        t = t_anim[frame_idx]
        tag = f"frame{frame_idx:05d}_t{t:.3f}s"

        # --- CSV ---
        df = pd.DataFrame({
            'drone_id': np.arange(n_drones),
            'x': traj[frame_idx, :, 0],
            'y': traj[frame_idx, :, 1],
            'z': traj[frame_idx, :, 2],
        })
        df.to_csv(os.path.join(export_dir, f"{tag}.csv"), index=False)

        # --- Scatterplot ---
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.set_xlabel('x  [m]', color='white')
        ax.set_ylabel('y  [m]', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
        ax.set_title(f't = {t:.3f} s', color='white', fontsize=11)

        # Trail
        start = max(0, frame_idx - trail_frames)
        for d in range(n_drones):
            ax.plot(
                traj[start:frame_idx + 1, d, 0],
                traj[start:frame_idx + 1, d, 1],
                lw=0.6, alpha=0.4, color=drone_colors[d],
            )

        ax.scatter(
            traj[frame_idx, :, 0],
            traj[frame_idx, :, 1],
            s=dot_size, c=drone_colors, zorder=3,
        )

        fig.savefig(os.path.join(export_dir, f"{tag}.png"),
                    dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  saved {tag}")

    print("Export done.")


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
        active_mask: (n_video_frames, n_drones)    – active drones
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


def load_npz(path, key):
    data = np.load(path, allow_pickle=False)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(data.keys())}")
    return data[key]


def _make_fig(trajectories, visibilities, dot_size: float):
    """Create figure, axes, scatter and return them together with axis limits."""
    active = visibilities.astype(bool)
    x_all = trajectories[:, :, 0][active]
    y_all = trajectories[:, :, 1][active]

    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="none")
    ax.set_facecolor("none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    scatter = ax.scatter([], [], s=dot_size, c="blue", linewidths=0)
    return fig, ax, scatter


def build_animation(trajectories, visibilities, fps: int, dot_size: float):
    """Return a FuncAnimation for the given trajectory and visibility arrays."""
    n_frames = trajectories.shape[0]
    fig, ax, scatter = _make_fig(trajectories, visibilities, dot_size)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return (scatter,)

    def update(frame_idx):
        visible = visibilities[frame_idx].astype(bool)
        pos = trajectories[frame_idx, visible, :2]
        scatter.set_offsets(pos)
        return (scatter,)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=1000 / fps,
        blit=True,
    )
    return fig, anim


def save_frames(trajectories, visibilities, dot_size: float, frames_dir: str):
    """Render and save each frame as a PNG in frames_dir."""
    os.makedirs(frames_dir, exist_ok=True)
    n_frames = trajectories.shape[0]
    fig, ax, scatter = _make_fig(trajectories, visibilities, dot_size)
    n_digits = len(str(n_frames - 1))
    print(f"Saving {n_frames} frames to {frames_dir} ...")
    for i in range(n_frames):
        visible = visibilities[i].astype(bool)
        pos = trajectories[i, visible, :2]
        scatter.set_offsets(pos if len(pos) else np.empty((0, 2)))
        fig.savefig(os.path.join(frames_dir, f"frame_{i:0{n_digits}d}.png"),
                    dpi=100, bbox_inches="tight", transparent=True)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_frames}", end="\r")
    print()
    plt.close(fig)

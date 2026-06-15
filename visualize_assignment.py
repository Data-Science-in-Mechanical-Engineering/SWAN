"""
Animate drone trajectories from a top-down (x, y) perspective.

Usage:
    python visualize_assignment.py \
        --simulation_data /path/to/simulation_results.npz \
        [--video /path/to/output.mp4] \
        [--export_dir /path/to/frames/] \
        [--ratio 0.3]

# uv run -m scripts.visualize_assignment --simulation_data /data/drone_show/whale/simulation_results.npz --video /data/drone_show/whale/assignment.mp4 --export_dir /data/drone_show/whale/frames/ --ratio 0.25
"""

import argparse
import os

import pandas as pd
import numpy as np
import scipy.interpolate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

TRAIL_FRAMES = 30   # how many past frames to draw as a fading trail


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animate drone trajectories – top-down (x, y) view."
    )
    parser.add_argument("--simulation_data", required=True,  help="Path to simulation_results.npz")
    parser.add_argument("--video",           default=None,   help="Path to output .mp4 file (optional)")
    parser.add_argument("--export_dir",      default=None,   help="Directory to write per-frame CSVs and PNGs (optional)")
    parser.add_argument("--ratio",  type=float, default=0.25,
                        help="Step between exported frames as fraction of total (e.g. 0.3 → frames at 0.0, 0.3, 0.6, 0.9)")
    parser.add_argument("--n_frames",   type=int,   default=300,  help="Number of animation frames")
    parser.add_argument("--fps",        type=int,   default=25,   help="Output frames per second")
    parser.add_argument("--dot_size",   type=float, default=20.0, help="Scatter dot area (matplotlib 's')")
    parser.add_argument("--trail",      type=int,   default=TRAIL_FRAMES,
                        help="Number of past frames shown as a trail per drone")
    args = parser.parse_args()

    if args.video is None and args.export_dir is None:
        parser.error("Provide at least one of --video or --export_dir.")

    traj, t_anim = load_and_prepare(
        args.simulation_data, n_anim_frames=args.n_frames
    )

    if args.video:
        make_animation(
            traj, t_anim,
            output_path=args.video,
            fps=args.fps,
            dot_size=args.dot_size,
            trail_frames=args.trail,
        )

    if args.export_dir:
        export_frames(
            traj, t_anim,
            export_dir=args.export_dir,
            ratio=args.ratio,
            dot_size=args.dot_size,
            trail_frames=args.trail,
        )

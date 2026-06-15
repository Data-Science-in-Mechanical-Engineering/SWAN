"""
Load trajectories.npz and visibilities.npz and animate the drone swarm.

Usage:
    python scripts/animate_tracking.py /path/to/folder [--fps 25] [--dot_size 20]

Loads trajectories.npz and visibilities.npz from the given folder and saves
the animation as video.mp4 in the same folder.

Expected array shapes
---------------------
trajectories : (n_frames, n_drones, D)  – D >= 2; first two dims used as x, y
visibilities : (n_frames, n_drones)     – bool / int (True = drone is visible)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


def main():
    parser = argparse.ArgumentParser(description="Animate drone trajectories from NPZ files.")
    parser.add_argument("folder", help="Folder containing trajectories.npz and visibilities.npz; video.mp4 is saved here.")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second (default: 25)")
    parser.add_argument("--dot_size", type=float, default=20, help="Scatter marker size (default: 20)")
    parser.add_argument("--traj_key", default="trajectories", help="Array key inside trajectories.npz")
    parser.add_argument("--vis_key", default="visibilities", help="Array key inside visibilities.npz")
    args = parser.parse_args()

    traj_path = os.path.join(args.folder, "trajectories.npz")
    vis_path  = os.path.join(args.folder, "visibilities.npz")
    out_path  = os.path.join(args.folder, "video.mp4")

    print(f"Loading trajectories from {traj_path} ...")
    trajectories = load_npz(traj_path, args.traj_key)
    print(f"  shape: {trajectories.shape}, dtype: {trajectories.dtype}")

    print(f"Loading visibilities from {vis_path} ...")
    visibilities = load_npz(vis_path, args.vis_key)
    print(f"  shape: {visibilities.shape}, dtype: {visibilities.dtype}")

    if trajectories.shape[:2] != visibilities.shape[:2]:
        raise ValueError(
            f"Shape mismatch: trajectories {trajectories.shape[:2]} vs visibilities {visibilities.shape[:2]}"
        )

    fig, anim = build_animation(trajectories, visibilities, fps=args.fps, dot_size=args.dot_size)

    print(f"Saving animation to {out_path} ...")
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=1800)
    anim.save(out_path, writer=writer)
    plt.close(fig)

    frames_dir = os.path.join(args.folder, "frames")
    save_frames(trajectories, visibilities, dot_size=args.dot_size, frames_dir=frames_dir)
    print("Done.")


if __name__ == "__main__":
    main()

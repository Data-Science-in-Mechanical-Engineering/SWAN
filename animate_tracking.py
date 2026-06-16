"""
Load trajectories.npz and visibilities.npz and animate the drone swarm.

Usage:
    python animate_tracking.py /path/to/folder [--fps 25] [--dot_size 20]

Loads trajectories.npz and visibilities.npz from the given folder and saves
the animation as video.mp4 in the same folder.

Expected array shapes
---------------------
trajectories : (n_frames, n_drones, D)  – D >= 2; first two dims used as x, y
visibilities : (n_frames, n_drones)     – bool / int (True = drone is visible)
"""

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from swan.utils import load_npz, build_animation, save_frames


if __name__ == "__main__":
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

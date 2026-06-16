"""
Animate drone trajectories from a top-down (x, y) perspective.

Usage:
    python visualize_assignment.py \
        --simulation_data /path/to/simulation_results.npz \
        [--video /path/to/output.mp4] \
        [--export_dir /path/to/frames/] \
        [--ratio 0.3]
"""

import argparse

from swan.utils import TRAIL_FRAMES, load_and_prepare, make_animation, export_frames


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

"""
Extract tracked drone positions, overlay them onto video frames, and save the result as a video.

Usage:
    python visualize_tracking.py \
        --simulation_data /path/to/simulation_results.npz \
        --trajectory_data /path/to/final_trajectories.npz \
        --video /path/to/input_video.mp4 \
        --output /path/to/output_video.mp4
"""

import argparse

from swan.utils import write_tracking_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay tracked drone positions onto a video.")
    parser.add_argument("--simulation_data", required=True, help="Path to simulation_results.npz")
    parser.add_argument("--trajectory_data", required=True, help="Path to final_trajectories.npz")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output video file")
    parser.add_argument("--dot_size", type=float, default=400, help="Marker size for drone dots")
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

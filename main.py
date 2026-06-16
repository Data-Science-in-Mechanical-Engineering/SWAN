#!/usr/bin/env python3
"""SWAN drone show pipeline - console application with interactive stages and Gradio UI."""

import argparse
import asyncio
import json
import os
import sys

import gradio as gr
import imageio.v3 as iio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Optional

from swan.video_generation import VideoGenerationConfig, generate_video
from swan.tracking import TrackingConfig, track_video
from swan.trajectory_generation import TrajectoryGenerationConfig, generate_trajectories
from swan.simulation import SimulationConfig, simulate_with_safety_filter
from visualize_tracking import write_tracking_video
from visualize_assignment import make_animation, export_frames
from animate_tracking import build_animation, save_frames


@dataclass
class PipelineState:
    n_drones: int = 500
    user_prompt: str = ""
    video_name: str = ""
    output_base_dir: str = ""
    current_stage: int = 0
    stage_outputs: dict = field(default_factory=dict)
    logs: str = ""
    running: bool = False
    complete: bool = False


def prompt_continue(stage_name: str, output_path: str, no_prompt: bool = False) -> bool:
    """Display result link and ask user if they want to continue."""
    print(f"\n{'='*60}")
    print(f"Stage '{stage_name}' complete!")
    print(f"Results available at: {output_path}")
    print(f"{'='*60}")
    
    if no_prompt:
        return True
    
    while True:
        response = input("Continue to next stage? [y/n]: ").strip().lower()
        if response in ('y', 'yes'):
            return True
        elif response in ('n', 'no'):
            return False
        print("Please enter 'y' or 'n'.")


def save_stage_result(output_base_dir: str, stage: str, data: dict):
    """Save stage result metadata for restart capability."""
    results_file = os.path.join(output_base_dir, "pipeline_results.json")
    results = {}
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    results[stage] = data
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def load_stage_result(output_base_dir: str, stage: str) -> dict | None:
    """Load stage result metadata if it exists."""
    results_file = os.path.join(output_base_dir, "pipeline_results.json")
    if not os.path.exists(results_file):
        return None
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results.get(stage)


async def run_video_generation(config: VideoGenerationConfig, user_prompt: str, output_base_dir: str) -> tuple[str, str, str]:
    """Run video generation stage and return (image_path, video_path, prompt_path)."""
    video_gen_dir = os.path.join(output_base_dir, "video_gen")
    os.makedirs(video_gen_dir, exist_ok=True)
    
    image_output_path = os.path.join(video_gen_dir, "image.png")
    video_output_path = os.path.join(video_gen_dir, "video.mp4")
    prompt_output_path = os.path.join(video_gen_dir, "prompt.json")
    
    await generate_video(
        config=config,
        user_prompt=user_prompt,
        image_output_path=image_output_path,
        video_output_path=video_output_path,
        prompt_output_path=prompt_output_path
    )
    
    return image_output_path, video_output_path, prompt_output_path


def run_tracking(video_path: str, prompt_path: str, output_base_dir: str, n_drones: int) -> tuple[str, str]:
    """Run tracking stage and return (tracking_visualization_path, tracking_points_only_path)."""
    tracking_output_dir = os.path.join(output_base_dir, "tracking")
    os.makedirs(tracking_output_dir, exist_ok=True)
    
    video = iio.imread(video_path)
    segmentation_prompt = json.load(open(prompt_path))["segmentation_prompt"]
    
    track_video(
        tracking_config=TrackingConfig(n_simultaneous_tracking_points=n_drones),
        segmentation_prompt=segmentation_prompt,
        video=video,
        output_dir=tracking_output_dir,
    )
    
    return (
        os.path.join(tracking_output_dir, "tracking_visualization.mp4"),
        os.path.join(tracking_output_dir, "tracking_points_only.mp4")
    )


def run_trajectory_generation(tracking_output_dir: str, trajectory_output_dir: str) -> tuple:
    """Run trajectory generation stage and return results."""
    os.makedirs(trajectory_output_dir, exist_ok=True)
    
    trajectory_generation_config = TrajectoryGenerationConfig()
    results = generate_trajectories(
        config=trajectory_generation_config,
        input_dir=tracking_output_dir,
        output_dir=trajectory_output_dir
    )
    
    trajectory_splines_final, t_frames_final, n_required_extra_drones, transformation_matrix = results
    
    if n_required_extra_drones > 0:
        print(f"Warning: {n_required_extra_drones} extra drones are required to execute the trajectories safely. "
              "Rerun tracking with a smaller n_simultaneous_tracking_points or adjust the safety parameters in the config.")
    
    return trajectory_splines_final, t_frames_final, n_required_extra_drones, transformation_matrix, trajectory_generation_config


def run_simulation(simulation_output_dir: str, trajectory_output_dir: str, trajectory_generation_config: TrajectoryGenerationConfig):
    """Run simulation and safety filter stage."""
    os.makedirs(simulation_output_dir, exist_ok=True)
    
    simulate_with_safety_filter(
        output_simulation_dir=simulation_output_dir,
        input_trajectory_dir=trajectory_output_dir,
        simulation_config=SimulationConfig(trajectory_generation_config=trajectory_generation_config, return_dense_trajectories=True)
    )


def run_visualization(simulation_output_dir: str, trajectory_output_dir: str, video_path: str, tracking_output_dir: str) -> tuple[str, str, str]:
    """Generate visualization videos and return paths to assignment and tracking videos."""
    from visualize_assignment import load_and_prepare

    simulation_data_path = os.path.join(simulation_output_dir, "simulation_results.npz")
    trajectory_data_path = os.path.join(trajectory_output_dir, "initial_trajectories.npz")

    # Generate assignment animation (top-down x, y view)
    assignment_video_path = os.path.join(simulation_output_dir, "assignment.mp4")
    export_dir = os.path.join(simulation_output_dir, "frames")

    traj, t_anim = load_and_prepare(simulation_data_path, n_anim_frames=300)
    
    make_animation(
        traj, t_anim,
        output_path=assignment_video_path,
        fps=25,
        dot_size=20.0,
        trail_frames=30,
    )

    export_frames(
        traj, t_anim,
        export_dir=export_dir,
        ratio=0.25,
        dot_size=20.0,
        trail_frames=30,
    )

    # Generate tracking overlay video
    tracking_video_path = os.path.join(simulation_output_dir, "tracking_overlay.mp4")
    write_tracking_video(
        simulation_path=simulation_data_path,
        trajectory_path=trajectory_data_path,
        video_path=video_path,
        output_path=tracking_video_path,
        dot_size=400,
        color=(255, 0, 0),
    )

    # Generate tracking animation from raw trajectories (top-down view)
    tracking_anim_path = os.path.join(simulation_output_dir, "tracking_animation.mp4")
    trajectories_path = os.path.join(tracking_output_dir, "trajectories.npz")
    visibilities_path = os.path.join(tracking_output_dir, "visibilities.npz")

    trajectories = np.load(trajectories_path, allow_pickle=False)["trajectories"]
    visibilities = np.load(visibilities_path, allow_pickle=False)["visibilities"]

    fig, anim = build_animation(trajectories, visibilities, fps=25, dot_size=20.0)
    os.makedirs(os.path.dirname(os.path.abspath(tracking_anim_path)), exist_ok=True)
    writer = matplotlib.animation.FFMpegWriter(fps=25, bitrate=1800)
    anim.save(tracking_anim_path, writer=writer)
    plt.close(fig)

    return assignment_video_path, tracking_video_path, tracking_anim_path


async def run_pipeline_async(state: PipelineState) -> PipelineState:
    """Run the full pipeline and update state with results."""
    output_base_dir = state.output_base_dir
    n_drones = state.n_drones
    user_prompt = state.user_prompt
    
    video_gen_dir = os.path.join(output_base_dir, "video_gen")
    tracking_output_dir = os.path.join(output_base_dir, "tracking")
    trajectory_output_dir = os.path.join(output_base_dir, "trajectory_generation")
    simulation_output_dir = os.path.join(output_base_dir, "simulation")
    
    # Stage 1: Video Generation
    state.current_stage = 1
    state.logs = "Running Stage 1: Video Generation...\n"
    state.running = True
    
    config = VideoGenerationConfig()
    image_path, video_path, prompt_path = await run_video_generation(
        config=config,
        user_prompt=user_prompt,
        output_base_dir=output_base_dir
    )
    save_stage_result(output_base_dir, "video", {
        "video_path": video_path,
        "prompt_path": prompt_path,
        "image_path": image_path
    })
    state.stage_outputs["video"] = video_path
    state.stage_outputs["image"] = image_path
    
    # Stage 2: Tracking
    state.current_stage = 2
    state.logs += "\nRunning Stage 2: Tracking...\n"
    
    tracking_viz_path, tracking_points_path = run_tracking(
        video_path=video_path,
        prompt_path=prompt_path,
        output_base_dir=output_base_dir,
        n_drones=n_drones
    )
    save_stage_result(output_base_dir, "tracking", {
        "visualization_path": tracking_viz_path,
        "points_only_path": tracking_points_path,
        "tracking_dir": tracking_output_dir
    })
    state.stage_outputs["tracking_video"] = tracking_viz_path
    state.stage_outputs["tracking_points"] = tracking_points_path
    
    # Stage 3: Trajectory Generation
    state.current_stage = 3
    state.logs += "\nRunning Stage 3: Trajectory Generation...\n"
    
    trajectory_splines_final, t_frames_final, n_required_extra_drones, transformation_matrix, trajectory_generation_config = run_trajectory_generation(
        tracking_output_dir=tracking_output_dir,
        trajectory_output_dir=trajectory_output_dir
    )
    save_stage_result(output_base_dir, "trajectory", {
        "n_required_extra_drones": n_required_extra_drones,
        "trajectory_dir": trajectory_output_dir
    })
    
    # Stage 4: Simulation
    state.current_stage = 4
    state.logs += "\nRunning Stage 4: Simulation and Safety Filter...\n"
    
    run_simulation(
        simulation_output_dir=simulation_output_dir,
        trajectory_output_dir=trajectory_output_dir,
        trajectory_generation_config=trajectory_generation_config
    )
    
    # Stage 5: Visualization
    state.current_stage = 5
    state.logs += "\nRunning Stage 5: Visualization...\n"
    
    assignment_video_path, tracking_video_path, tracking_anim_path = run_visualization(
        simulation_output_dir=simulation_output_dir,
        trajectory_output_dir=trajectory_output_dir,
        video_path=video_path,
        tracking_output_dir=tracking_output_dir
    )
    save_stage_result(output_base_dir, "visualization", {
        "assignment_video_path": assignment_video_path,
        "tracking_video_path": tracking_video_path,
        "tracking_anim_path": tracking_anim_path
    })
    
    state.stage_outputs["assignment"] = assignment_video_path
    state.stage_outputs["tracking_overlay"] = tracking_video_path
    state.stage_outputs["tracking_animation"] = tracking_anim_path
    state.running = False
    state.complete = True
    state.logs += "\nPipeline complete! All outputs generated."
    
    return state


def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks() as demo:
        pipeline_state = gr.State(PipelineState())
        
        with gr.Accordion("Configuration", open=True):
            show_name = gr.Textbox(label="Show Name", value="my_drone_show", placeholder="Enter a name for your drone show")
            user_prompt = gr.Textbox(
                label="User Prompt", 
                value="A T-Rex roaring and turning its head from left to right standing on a large flat rock.",
                lines=4,
                placeholder="Describe the drone show you want to create"
            )
            n_drones = gr.Slider(50, 2000, value=500, step=50, label="Number of Drones")
            
            run_btn = gr.Button("Run Pipeline", variant="primary")
            reset_btn = gr.Button("Reset", variant="secondary")

            logs_output = gr.Textbox(label="Logs", lines=10, interactive=False)
        
        with gr.Accordion("Video Generation", open=True):
            video_output = gr.Video(label="Generated Video", format="mp4")
        
        with gr.Accordion("Tracking", open=True):
            with gr.Row():
                tracking_video = gr.Video(label="Points on Video", format="mp4")
            with gr.Row():
                points_only_video = gr.Video(label="Points Only", format="mp4")
        
        
        with gr.Accordion("Final Show", open=True):
            assignment_video = gr.Video(label="Assignment View (Top-down)", format="mp4")
            tracking_overlay = gr.Video(label="Tracking Overlay", format="mp4")
        
        async def run_pipeline_ui(show_name_val, prompt_val, drones_val, state):
            """Run the full pipeline locally, streaming results after each stage."""
            state.n_drones = int(drones_val)
            state.user_prompt = prompt_val
            state.video_name = show_name_val.replace(" ", "_").replace("/", "_")
            state.output_base_dir = f"/app/out/{state.video_name}_{state.n_drones}_drones"
            state.current_stage = 0
            state.stage_outputs = {}
            state.logs = ""
            state.running = True
            state.complete = False

            video_path = None
            tracking_viz_path = None
            tracking_points_path = None
            assignment_video_path = None
            tracking_video_path = None
            
            # Stage 1: Video Generation
            try:
                config = VideoGenerationConfig()
                image_path, video_path, prompt_path = await run_video_generation(
                    config=config,
                    user_prompt=state.user_prompt,
                    output_base_dir=state.output_base_dir,
                )
                save_stage_result(state.output_base_dir, "video", {
                    "video_path": video_path,
                    "prompt_path": prompt_path,
                    "image_path": image_path,
                })
                state.stage_outputs["video"] = video_path
                state.stage_outputs["image"] = image_path
                state.logs = "Stage 1: Video Generation complete.\n"
                # Yield after video generation
                yield (
                    state,
                    video_path,
                    None,
                    None,
                    None,
                    None,
                    state.logs,
                )
            except Exception as e:
                state.running = False
                state.logs = f"Pipeline error during video generation: {e}"
                yield (
                    state,
                    None,
                    None,
                    None,
                    None,
                    None,
                    state.logs,
                )
                return
            
            # Stage 2: Tracking
            try:
                tracking_viz_path, tracking_points_path = run_tracking(
                    video_path=video_path,
                    prompt_path=prompt_path,
                    output_base_dir=state.output_base_dir,
                    n_drones=state.n_drones,
                )
                tracking_output_dir = os.path.join(state.output_base_dir, "tracking")
                save_stage_result(state.output_base_dir, "tracking", {
                    "visualization_path": tracking_viz_path,
                    "points_only_path": tracking_points_path,
                    "tracking_dir": tracking_output_dir,
                })
                state.stage_outputs["tracking_video"] = tracking_viz_path
                state.stage_outputs["tracking_points"] = tracking_points_path
                state.logs += "Stage 2: Tracking complete.\n"
                yield (
                    state,
                    video_path,
                    tracking_viz_path,
                    tracking_points_path,
                    None,
                    None,
                    state.logs,
                )
            except Exception as e:
                state.running = False
                state.logs += f"\nPipeline error during tracking: {e}"
                yield (
                    state,
                    video_path,
                    None,
                    None,
                    None,
                    None,
                    state.logs,
                )
                return
            
            # Stage 3: Trajectory Generation
            try:
                trajectory_splines_final, t_frames_final, n_required_extra_drones, transformation_matrix, trajectory_generation_config = run_trajectory_generation(
                    tracking_output_dir=os.path.join(state.output_base_dir, "tracking"),
                    trajectory_output_dir=os.path.join(state.output_base_dir, "trajectory_generation"),
                )
                save_stage_result(state.output_base_dir, "trajectory", {
                    "n_required_extra_drones": n_required_extra_drones,
                    "trajectory_dir": os.path.join(state.output_base_dir, "trajectory_generation"),
                })
                state.logs += "Stage 3: Trajectory Generation complete.\n"
                # No visual output for this stage yet, just update logs
                yield (
                    state,
                    video_path,
                    tracking_viz_path,
                    tracking_points_path,
                    None,
                    None,
                    state.logs,
                )
            except Exception as e:
                state.running = False
                state.logs += f"\nPipeline error during trajectory generation: {e}"
                yield (
                    state,
                    video_path,
                    tracking_viz_path,
                    tracking_points_path,
                    None,
                    None,
                    state.logs,
                )
                return
            
            # Stage 4: Simulation
            try:
                run_simulation(
                    simulation_output_dir=os.path.join(state.output_base_dir, "simulation"),
                    trajectory_output_dir=os.path.join(state.output_base_dir, "trajectory_generation"),
                    trajectory_generation_config=trajectory_generation_config,
                )
                state.logs += "Stage 4: Simulation complete.\n"
                yield (
                    state,
                    video_path,
                    tracking_viz_path,
                    tracking_points_path,
                    None,
                    None,
                    state.logs,
                )
            except Exception as e:
                state.running = False
                state.logs += f"\nPipeline error during simulation: {e}"
                yield (
                    state,
                    video_path,
                    tracking_viz_path,
                    tracking_points_path,
                    None,
                    None,
                    state.logs,
                )
                return
            
            # Stage 5: Visualization
            try:
                assignment_video_path, tracking_video_path, tracking_anim_path = run_visualization(
                    simulation_output_dir=os.path.join(state.output_base_dir, "simulation"),
                    trajectory_output_dir=os.path.join(state.output_base_dir, "trajectory_generation"),
                    video_path=video_path,
                    tracking_output_dir=os.path.join(state.output_base_dir, "tracking"),
                )
                save_stage_result(state.output_base_dir, "visualization", {
                    "assignment_video_path": assignment_video_path,
                    "tracking_video_path": tracking_video_path,
                    "tracking_anim_path": tracking_anim_path,
                })
                state.stage_outputs["assignment"] = assignment_video_path
                state.stage_outputs["tracking_overlay"] = tracking_video_path
                state.stage_outputs["tracking_animation"] = tracking_anim_path
                state.logs += "Stage 5: Visualization complete.\n"
                yield (
                    state,
                    video_path,
                    tracking_viz_path,
                    tracking_points_path,
                    assignment_video_path,
                    tracking_video_path,
                    state.logs,
                )
            except Exception as e:
                state.running = False
                state.logs += f"\nPipeline error during visualization: {e}"
                yield (
                    state,
                    video_path,
                    tracking_viz_path,
                    tracking_points_path,
                    None,
                    None,
                    state.logs,
                )
                return
            
            # Final completion
            state.running = False
            state.complete = True
            state.logs += "Pipeline complete! All outputs generated.\n"
            yield (
                state,
                video_path,
                tracking_viz_path,
                tracking_points_path,
                assignment_video_path,
                tracking_video_path,
                state.logs,
            )
        
        def reset_all():
            return (
                PipelineState(),
                None,
                None,
                None,
                None,
                None,
                "",
            )
        
        run_btn.click(
            fn=run_pipeline_ui,
            inputs=[show_name, user_prompt, n_drones, pipeline_state],
            outputs=[
                pipeline_state,
                video_output,
                tracking_video,
                points_only_video,
                assignment_video,
                tracking_overlay,
                logs_output,
            ],
            queue=True
        )
        
        reset_btn.click(
            fn=reset_all,
            outputs=[
                pipeline_state,
                video_output,
                tracking_video,
                points_only_video,
                assignment_video,
                tracking_overlay,
                logs_output,
            ]
        )
    
    return demo


async def main_async():
    parser = argparse.ArgumentParser(description="SWAN drone show pipeline")
    parser.add_argument("--n-drones", type=int, default=500, help="Number of drones")
    parser.add_argument("--video-name", type=str, default="trex", help="Video name for output directory")
    parser.add_argument("--user-prompt", type=str, default="A T-Rex roaring and turning its head from left to right standing on a large flat rock.",
                        help="User prompt for video generation")
    parser.add_argument("--start-stage", type=str, choices=["video", "segmentation", "tracking", "trajectory", "simulation", "visualization"],
                        help="Start from a specific stage (skip earlier stages)")
    parser.add_argument("--video-path", type=str, help="Path to existing video (for restart after video generation)")
    parser.add_argument("--prompt-path", type=str, help="Path to existing prompt.json (for restart after video generation)")
    parser.add_argument("--tracking-dir", type=str, help="Path to tracking output directory (for restart after tracking)")
    parser.add_argument("--no-prompt", action="store_true", help="Run without interactive prompts (for UI/automation)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override (for UI/automation)")
    parser.add_argument("--cli", action="store_true", help="Run in console mode (CLI only, no UI)")
    
    args = parser.parse_args()
    
    if args.output_dir:
        output_base_dir = args.output_dir
    else:
        output_base_dir = f"/app/out/{args.video_name}_{args.n_drones}_drones"
    os.makedirs(output_base_dir, exist_ok=True)
    
    video_gen_dir = os.path.join(output_base_dir, "video_gen")
    tracking_output_dir = os.path.join(output_base_dir, "tracking")
    trajectory_output_dir = os.path.join(output_base_dir, "trajectory_generation")
    simulation_output_dir = os.path.join(output_base_dir, "simulation")
    
    # Determine starting point
    start_stage = args.start_stage
    
    # Stage 1: Video Generation
    if start_stage is None or start_stage == "video":
        print("Running Stage 1: Video Generation...")
        config = VideoGenerationConfig()
        image_path, video_path, prompt_path = await run_video_generation(
            config=config,
            user_prompt=args.user_prompt,
            output_base_dir=output_base_dir
        )
        save_stage_result(output_base_dir, "video", {
            "video_path": video_path,
            "prompt_path": prompt_path,
            "image_path": image_path
        })
        
        if not prompt_continue("video_generation", f"file://{video_path}", no_prompt=args.no_prompt):
            return
    else:
        # Load from existing results or use provided paths
        video_result = load_stage_result(output_base_dir, "video")
        if video_result:
            video_path = video_result["video_path"]
            prompt_path = video_result["prompt_path"]
            image_path = video_result["image_path"]
        elif args.video_path and args.prompt_path:
            video_path = args.video_path
            prompt_path = args.prompt_path
            image_path = video_path.replace(".mp4", ".png")
        else:
            print("Error: No video results found and no --video-path/--prompt-path provided.")
            return
    
    # Stage 2 & 3: Tracking (includes segmentation)
    if start_stage is None or start_stage in ("segmentation", "tracking"):
        print("\nRunning Stage 2 & 3: Segmentation and Tracking...")
        tracking_viz_path, tracking_points_path = run_tracking(
            video_path=video_path,
            prompt_path=prompt_path,
            output_base_dir=output_base_dir,
            n_drones=args.n_drones
        )
        save_stage_result(output_base_dir, "tracking", {
            "visualization_path": tracking_viz_path,
            "points_only_path": tracking_points_path,
            "tracking_dir": tracking_output_dir
        })
        
        if not prompt_continue("tracking", f"file://{tracking_viz_path}", no_prompt=args.no_prompt):
            return
    else:
        tracking_result = load_stage_result(output_base_dir, "tracking")
        if tracking_result:
            tracking_output_dir = tracking_result.get("tracking_dir", tracking_output_dir)
        elif args.tracking_dir:
            tracking_output_dir = args.tracking_dir
        else:
            print("Error: No tracking results found and no --tracking-dir provided.")
            return
    
    # Stage 4: Trajectory Generation
    if start_stage is None or start_stage == "trajectory":
        print("\nRunning Stage 4: Trajectory Generation...")
        trajectory_splines_final, t_frames_final, n_required_extra_drones, transformation_matrix, trajectory_generation_config = run_trajectory_generation(
            tracking_output_dir=tracking_output_dir,
            trajectory_output_dir=trajectory_output_dir
        )
        save_stage_result(output_base_dir, "trajectory", {
            "n_required_extra_drones": n_required_extra_drones,
            "trajectory_dir": trajectory_output_dir
        })
        
        if not prompt_continue("trajectory_generation", f"file://{trajectory_output_dir}", no_prompt=args.no_prompt):
            return
    else:
        traj_result = load_stage_result(output_base_dir, "trajectory")
        if traj_result:
            trajectory_generation_config = TrajectoryGenerationConfig()
        else:
            print("Error: No trajectory results found. Cannot continue from this stage.")
            return
    
    # Stage 5: Simulation and Safety Filter
    if start_stage is None or start_stage == "simulation":
        print("\nRunning Stage 5: Simulation and Safety Filter...")
        run_simulation(
            simulation_output_dir=simulation_output_dir,
            trajectory_output_dir=trajectory_output_dir,
            trajectory_generation_config=trajectory_generation_config
        )
        
        if not prompt_continue("simulation", f"file://{simulation_output_dir}", no_prompt=args.no_prompt):
            return
    
    # Stage 6: Visualization
    if start_stage is None or start_stage == "visualization":
        print("\nRunning Stage 6: Visualization...")
        assignment_video_path, tracking_video_path, tracking_anim_path = run_visualization(
            simulation_output_dir=simulation_output_dir,
            trajectory_output_dir=trajectory_output_dir,
            video_path=video_path,
            tracking_output_dir=tracking_output_dir
        )
        save_stage_result(output_base_dir, "visualization", {
            "assignment_video_path": assignment_video_path,
            "tracking_video_path": tracking_video_path,
            "tracking_anim_path": tracking_anim_path
        })

        print(f"\n{'='*60}")
        print("Pipeline complete!")
        print(f"Assignment video: file://{assignment_video_path}")
        print(f"Tracking overlay video: file://{tracking_video_path}")
        print(f"Tracking animation: file://{tracking_anim_path}")
        print(f"Pipeline state saved to: {output_base_dir}/pipeline_results.json")
        print("Use --start-stage to resume from any stage.")
        print(f"{'='*60}")


def main():
    # Check if --cli flag is set
    if "--cli" in sys.argv:
        asyncio.run(main_async())
    else:
        demo = create_interface()
        demo.launch(server_port=7860, share=False, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
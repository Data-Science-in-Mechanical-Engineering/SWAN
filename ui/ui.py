#!/usr/bin/env python3
"""Gradio UI for SWAN drone show pipeline.

Supports local execution and SSH-based remote execution.
"""

# Make the project root discoverable when this module is executed directly from
# `ui/ui.py`.
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr

from swan.pipeline import (
    PipelineState,
    VideoGenerationConfig,
    run_video_generation,
    run_tracking,
    run_trajectory_generation,
    run_simulation,
    run_visualization,
    save_stage_result,
)

def _create_local_interface():
    """Build the Gradio interface for local pipeline execution."""

    with gr.Blocks() as demo:
        pipeline_state = gr.State(PipelineState())

        with gr.Row():
            with gr.Column(scale=3):
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

            with gr.Column(scale=1):
                logs_output = gr.Textbox(label="Logs", lines=25, interactive=False)

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


def create_interface():
    """Create the Gradio interface.

    Parameters
    ----------
    mode : {"local", "ssh"}
        ``local`` runs the pipeline in-process, ``ssh`` drives a remote Docker
        container over SSH.
    """
    return _create_local_interface()

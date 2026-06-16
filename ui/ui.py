#!/usr/bin/env python3
"""Gradio UI for SWAN drone show pipeline.

Supports local execution with explicit user confirmation between pipeline
stages. Video generation runs first, then the user reviews the result and
clicks continue to advance to tracking, and so on.
"""

# Make the project root discoverable when this module is executed directly from
# `ui/ui.py`.
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr
import shutil

from swan.pipeline import (
    PipelineState,
    VideoGenerationConfig,
    run_video_generation,
    run_segmentation,
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

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                with gr.Accordion("Configuration", open=True):
                    show_name = gr.Textbox(
                        label="Show Name",
                        value="my_drone_show",
                        placeholder="Enter a name for your drone show",
                    )
                    user_prompt = gr.Textbox(
                        label="User Prompt",
                        value=(
                            "A T-Rex roaring and turning its head from left to right "
                            "standing on a large flat rock."
                        ),
                        lines=4,
                        placeholder="Describe the drone show you want to create",
                    )
                    n_drones = gr.Slider(
                        50, 2000, value=500, step=50, label="Number of Drones"
                    )
                    with gr.Row():
                        generate_video_btn = gr.Button("Generate Video", variant="primary")
                        reset_btn = gr.Button("Reset", variant="secondary")

            with gr.Column(scale=1):
                logs_output = gr.Textbox(label="Logs", lines=17, interactive=False)

        with gr.Accordion("Video Generation", open=True):
            video_output = gr.Video(label="Generated Video", format="mp4")
            continue_segmentation_btn = gr.Button(
                "Continue to Segmentation", variant="primary", interactive=False
            )

        with gr.Accordion("Segmentation", open=True):
            with gr.Row():
                segmentation_video = gr.Video(
                    label="Segmentation Overlay", format="mp4"
                )
            continue_tracking_btn = gr.Button(
                "Continue to Tracking", variant="primary", interactive=False
            )

        with gr.Accordion("Tracking", open=True):
            with gr.Row():
                tracking_video = gr.Video(label="Points on Video", format="mp4")
            with gr.Row():
                points_only_video = gr.Video(
                    label="Points Only", format="mp4"
                )
            continue_final_btn = gr.Button(
                "Continue to Final Stages", variant="primary", interactive=False
            )

        with gr.Accordion("Final Show", open=True):
            assignment_video = gr.Video(
                label="Assignment View (Top-down)", format="mp4"
            )
            tracking_overlay = gr.Video(
                label="Tracking Overlay", format="mp4"
            )

        async def run_video_generation_ui(show_name_val, prompt_val, drones_val, state):
            """Run the first stage and present the generated video."""
            state.n_drones = int(drones_val)
            state.user_prompt = prompt_val
            state.video_name = show_name_val.replace(" ", "_").replace("/", "_")
            state.output_base_dir = (
                f"/app/out/{state.video_name}_{state.n_drones}_drones"
            )
            state.current_stage = 0
            state.stage_outputs = {}
            state.logs = "Stage 1: Video Generation started...\n"
            state.running = True
            state.complete = False

            yield (
                state,
                None,
                gr.update(interactive=False),
                gr.update(interactive=False),
                state.logs,
            )

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
                state.stage_outputs["prompt_path"] = prompt_path
                state.current_stage = 1
                state.logs += (
                    "Stage 1: Video Generation complete.\n"
                    "Review the video and click 'Continue to Segmentation'."
                )
                yield (
                    state,
                    video_path,
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    state.logs,
                )
            except Exception as e:
                state.running = False
                state.logs = f"Pipeline error during video generation: {e}"
                yield (
                    state,
                    None,
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    state.logs,
                )

        def run_segmentation_ui(state):
            """Run segmentation after the user confirms the generated video."""
            if state.current_stage < 1 or "video" not in state.stage_outputs:
                state.logs += "\nCannot run segmentation: no video available."
                yield state, None, gr.update(interactive=False), gr.update(interactive=False), state.logs
                return

            video_path = state.stage_outputs["video"]
            prompt_path = state.stage_outputs.get("prompt_path")

            state.running = True
            state.logs += "\nStage 2: Segmentation started...\n"
            yield (
                state,
                None,
                gr.update(interactive=False),
                gr.update(interactive=False),
                state.logs,
            )

            try:
                segmentation_overlay_path = run_segmentation(
                    video_path=video_path,
                    prompt_path=prompt_path,
                    output_base_dir=state.output_base_dir,
                )
                tracking_output_dir = os.path.join(state.output_base_dir, "tracking")
                save_stage_result(state.output_base_dir, "segmentation", {
                    "segmentation_overlay_path": segmentation_overlay_path,
                    "tracking_dir": tracking_output_dir,
                })
                state.stage_outputs["segmentation_overlay"] = segmentation_overlay_path
                state.current_stage = 2
                state.logs += (
                    "Stage 2: Segmentation complete.\n"
                    "Review the overlay and click 'Continue to Tracking'."
                )
                yield (
                    state,
                    segmentation_overlay_path,
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    state.logs,
                )
            except Exception as e:
                state.running = False
                state.logs += f"\nPipeline error during segmentation: {e}"
                yield state, None, gr.update(interactive=True), gr.update(interactive=False), state.logs

        def run_tracking_ui(state):
            """Run tracking after the user confirms the segmentation overlay."""
            if state.current_stage < 2 or "video" not in state.stage_outputs:
                state.logs += "\nCannot run tracking: segmentation not complete."
                yield state, None, None, gr.update(interactive=False), gr.update(interactive=False), state.logs
                return

            video_path = state.stage_outputs["video"]
            prompt_path = state.stage_outputs.get("prompt_path")

            state.running = True
            state.logs += "\nStage 3: Tracking started...\n"
            yield (
                state,
                None,
                None,
                gr.update(interactive=False),
                gr.update(interactive=False),
                state.logs,
            )

            try:
                tracking_viz_path, tracking_points_path, segmentation_overlay_path = run_tracking(
                    video_path=video_path,
                    prompt_path=prompt_path,
                    output_base_dir=state.output_base_dir,
                    n_drones=state.n_drones,
                )
                tracking_output_dir = os.path.join(state.output_base_dir, "tracking")
                save_stage_result(state.output_base_dir, "tracking", {
                    "visualization_path": tracking_viz_path,
                    "points_only_path": tracking_points_path,
                    "segmentation_overlay_path": segmentation_overlay_path,
                    "tracking_dir": tracking_output_dir,
                })
                state.stage_outputs["tracking_video"] = tracking_viz_path
                state.stage_outputs["tracking_points"] = tracking_points_path
                state.stage_outputs["segmentation_overlay"] = segmentation_overlay_path
                state.current_stage = 3
                state.logs += (
                    "Stage 3: Tracking complete.\n"
                    "Review the results and click 'Continue to Final Stages'."
                )
                yield (
                    state,
                    tracking_viz_path,
                    tracking_points_path,
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    state.logs,
                )
            except Exception as e:
                state.running = False
                state.logs += f"\nPipeline error during tracking: {e}"
                yield (
                    state,
                    None,
                    None,
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    state.logs,
                )

        async def run_final_stages_ui(state):
            """Run trajectory generation, simulation, and visualization."""
            if state.current_stage < 3:
                state.logs += "\nCannot run final stages: tracking not complete."
                yield state, None, None, gr.update(interactive=False), state.logs
                return

            state.running = True
            state.logs += "\nStage 4: Trajectory Generation started...\n"
            yield state, None, None, gr.update(interactive=False), state.logs

            try:
                (
                    trajectory_splines_final,
                    t_frames_final,
                    n_required_extra_drones,
                    transformation_matrix,
                    trajectory_generation_config,
                ) = run_trajectory_generation(
                    tracking_output_dir=os.path.join(
                        state.output_base_dir, "tracking"
                    ),
                    trajectory_output_dir=os.path.join(
                        state.output_base_dir, "trajectory_generation"
                    ),
                )
                save_stage_result(state.output_base_dir, "trajectory", {
                    "n_required_extra_drones": n_required_extra_drones,
                    "trajectory_dir": os.path.join(
                        state.output_base_dir, "trajectory_generation"
                    ),
                })
                state.logs += "Stage 4: Trajectory Generation complete.\n"
                if n_required_extra_drones > 0:
                    state.logs += (
                        f"Note: {n_required_extra_drones} extra drones are required "
                        "to execute the trajectories safely.\n"
                    )
                yield state, None, None, gr.update(interactive=False), state.logs
            except Exception as e:
                state.running = False
                state.logs += f"\nPipeline error during trajectory generation: {e}"
                yield state, None, None, gr.update(interactive=False), state.logs
                return

            try:
                state.logs += "Stage 5: Simulation and Safety Filter started...\n"
                yield state, None, None, gr.update(interactive=False), state.logs
                run_simulation(
                    simulation_output_dir=os.path.join(
                        state.output_base_dir, "simulation"
                    ),
                    trajectory_output_dir=os.path.join(
                        state.output_base_dir, "trajectory_generation"
                    ),
                    trajectory_generation_config=trajectory_generation_config,
                )
                state.logs += "Stage 5: Simulation and Safety Filter complete.\n"
                yield state, None, None, gr.update(interactive=False), state.logs
            except Exception as e:
                state.running = False
                state.logs += f"\nPipeline error during simulation: {e}"
                yield state, None, None, gr.update(interactive=False), state.logs
                return

            try:
                state.logs += "Stage 6: Visualization started...\n"
                yield state, None, None, gr.update(interactive=False), state.logs
                assignment_video_path, tracking_video_path, tracking_anim_path = (
                    run_visualization(
                        simulation_output_dir=os.path.join(
                            state.output_base_dir, "simulation"
                        ),
                        trajectory_output_dir=os.path.join(
                            state.output_base_dir, "trajectory_generation"
                        ),
                        video_path=state.stage_outputs["video"],
                        tracking_output_dir=os.path.join(
                            state.output_base_dir, "tracking"
                        ),
                    )
                )
                save_stage_result(state.output_base_dir, "visualization", {
                    "assignment_video_path": assignment_video_path,
                    "tracking_video_path": tracking_video_path,
                    "tracking_anim_path": tracking_anim_path,
                })
                state.stage_outputs["assignment"] = assignment_video_path
                state.stage_outputs["tracking_overlay"] = tracking_video_path
                state.stage_outputs["tracking_animation"] = tracking_anim_path
                state.current_stage = 5
                state.running = False
                state.complete = True
                state.logs += (
                    "Stage 5: Visualization complete.\n"
                    "Pipeline complete! All outputs generated."
                )
                yield (
                    state,
                    assignment_video_path,
                    tracking_video_path,
                    gr.update(interactive=True),
                    state.logs,
                )
            except Exception as e:
                state.running = False
                state.logs += f"\nPipeline error during visualization: {e}"
                yield state, None, None, gr.update(interactive=True), state.logs
                return

        def reset_all(state):
             if state.output_base_dir and os.path.exists(state.output_base_dir):
                 shutil.rmtree(state.output_base_dir)
             return (
                 PipelineState(),
                 None,                          # video_output
                 gr.update(interactive=False),  # continue_segmentation_btn
                 None,                          # segmentation_video
                 gr.update(interactive=False),  # continue_tracking_btn
                 None,                          # tracking_video
                 None,                          # points_only_video
                 gr.update(interactive=False),  # continue_final_btn
                 None,                          # assignment_video
                 None,                          # tracking_overlay
                 "",                            # logs_output
             )

        generate_video_btn.click(
             fn=run_video_generation_ui,
             inputs=[show_name, user_prompt, n_drones, pipeline_state],
             outputs=[
                 pipeline_state,
                 video_output,
                 continue_segmentation_btn,
                 generate_video_btn,
                 logs_output,
             ],
             queue=True,
         )

        continue_segmentation_btn.click(
             fn=run_segmentation_ui,
             inputs=[pipeline_state],
             outputs=[
                 pipeline_state,
                 segmentation_video,
                 continue_segmentation_btn,
                 continue_tracking_btn,
                 logs_output,
             ],
             queue=True,
         )

        continue_tracking_btn.click(
             fn=run_tracking_ui,
             inputs=[pipeline_state],
             outputs=[
                 pipeline_state,
                 tracking_video,
                 points_only_video,
                 continue_tracking_btn,
                 continue_final_btn,
                 logs_output,
             ],
             queue=True,
         )

        continue_final_btn.click(
             fn=run_final_stages_ui,
             inputs=[pipeline_state],
             outputs=[
                 pipeline_state,
                 assignment_video,
                 tracking_overlay,
                 continue_final_btn,
                 logs_output,
             ],
             queue=True,
         )

        reset_btn.click(
             fn=reset_all,
             inputs=[pipeline_state],
             outputs=[
                 pipeline_state,
                 video_output,
                 continue_segmentation_btn,
                 segmentation_video,
                 continue_tracking_btn,
                 tracking_video,
                 points_only_video,
                 continue_final_btn,
                 assignment_video,
                 tracking_overlay,
                 logs_output,
             ],
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

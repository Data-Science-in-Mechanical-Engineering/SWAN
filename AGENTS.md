# Project Name

The SWAN project is a demo that can create drone shows for thousands of drones from text prompts.
It has the following pipeline stages:
1. Video generation with a video generating model (e.g., Wan 2.2)
2. Segmentation of the main object (integrated in tracking stage for UI, separate for CLI)
3. Tracking of points inside the main object.
4. Trajectory generation (assignments via optimal flow + assembling with takeoff/landing)
5. Safety filter via AXSwarm
6. Visualization of results (top-down assignment and tracking overlay videos)

## Code Style

- Use standard Python conventions (PEP 8)

## Architecture

The SWAN pipeline is orchestrated from `main.py` (CLI/local UI entry point) and implemented as a set of modular stages under the `swan/` package.

### Runtime environment

- The project is designed to run inside the Docker image defined by `Dockerfile`, which is based on `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` and uses a UV-managed Python 3.12 virtual environment at `/opt/venv`.
- `docker-compose.yaml` mounts the repository root into `/app`, mounts `./weights` and `./comfyui_models` for model storage, and passes through all NVIDIA GPUs.
- `PYTHONPATH` is set to `/app/swan` so the package imports work without installation.

### Pipeline stages

1. **Video generation** (`swan/video_generation.py`)
    - Expands the user prompt with a local LLM (`PromptExpander`) into cinematic components and a segmentation prompt.
    - Generates a start frame and a follow-up video via ComfyUI using the `ComfyUIServer` / `ComfyUIClient` wrappers around the ComfyUI HTTP API.
    - Workflow templates are filled using presets defined in `WorkflowTemplate` and executed asynchronously.

2. **Segmentation** (`swan/tracking.py`, integrated in tracking stage)
    - Segments the main object per frame using LangSAM + SAM 2.1 and caches results.
    - CLI mode runs segmentation separately; UI mode integrates it with tracking.

3. **Tracking** (`swan/tracking.py`)
    - Samples initial drone positions via Centroidal Voronoi Tessellation (K-Means) inside the segmentation mask.
    - Tracks points through the video with CoTracker3.
    - Applies a force field to keep points inside the mask, avoids overcrowding, removes outliers, and heals lost tracks.
    - Outputs per-frame 2D trajectories, visibility flags, and segment-start flags.

4. **Trajectory generation** (`swan/trajectory_generation.py`)
    - Loads image-space tracking results, removes short segments, and reorders drones.
    - Densifies and smooths trajectories with smoothing splines, then rescales image-space units to real-world meters so inter-drone distances satisfy a safety margin.
    - Splits non-continuous logical drone segments and solves a min-cost-flow problem (`networkx`) to assign segments to the minimum number of physical drones, adding extra drones where kinematically necessary.
    - Computes 3D takeoff / landing pads on a ground grid and generates quintic takeoff, landing, and transition trajectories.
    - Routes transitions collision-free with X-axis avoidance bumps and finalizes quintic B-splines at the requested output frequency.
    - Writes `initial_trajectories.npz` with splines, actions, timestamps, lead-in / lead-out frames, and the image-to-world transformation matrix.

5. **Simulation / safety filter** (`swan/simulation.py`)
    - Loads the generated splines and parameters, then runs the AXSwarm MPC solver inside a `crazyflow` simulation.
    - Uses a custom `CollisionlessSim` subclass to disable contact physics and avoid excessive memory use for large swarms.
    - Checks the resulting dense trajectories for collisions.
    - Saves `simulation_results.npz` with trajectories_simulated, t_frames, n_leadin_frames, n_leadout_frames.

6. **Visualization**
    - Visualization helpers are consolidated in `swan/utils.py`:
        - `make_animation()` / `export_frames()` – top-down (x, y) assignment animation and per-frame exports.
        - `write_tracking_video()` – overlay simulated drones onto the original video.
        - `build_animation()` / `save_frames()` – raw image-space tracking animation.
        - `write_mask_overlay_video()` – renders segmentation masks as transparent video overlay.
    - `swan/pipeline.py` wires these helpers together in `run_visualization()`.
    - `main.py` exposes the local Gradio UI; the SSH-based remote UI lives in `ui/ui.py`.
    - Generates `assignment.mp4`, per-frame CSV/PNG exports, `tracking_overlay.mp4`, and `tracking_animation.mp4`.

### Utilities

- `swan/utils.py` provides helpers for rendering tracking visualizations and assigning per-drone colors.

### UI
- `ui/ui.py` provides the Gradio web interface with interactive stage-by-stage execution and result previews.

### Models and assets

- Model weights are downloaded separately with `download_weights.py`. It supports downloading only the core vision models (CoTracker3, SAM 2.1, Grounding DINO) or everything including the ComfyUI video models.
- ComfyUI is included as a Git submodule and mounted at `/app/comfyui`.
- Static data such as prompts, workflow JSONs, and AXSwarm settings live under `static_data/`.
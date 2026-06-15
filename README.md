# SWAN - Swarm Animation From Natural Language
SWAN is generative AI pipeline that creates dynamic choreographies for drone light shows with thousands of UAVs, based on natural language descriptions.

## Setup

```bash
# Clone the repository with the --recursive flag to include the comfyui submodule
git clone --recursive https://github.com/SWAN-URL/repo-name.git
```

Models will be cached in `./weights` and `./comfyui_models` on your host filesystem.

## Usage

```bash
# Build the image
docker build -t swan .

# Start the container (requires Docker Compose v2)
docker compose up -d

# Enter the container shell
docker compose exec swan bash
```

Inside the container:

```bash
# Download model weights (choose 1 for core models only, or 2 for all including video models)
python download_weights.py

# Launch Jupyter notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

Then open `http://localhost:8888` in your browser and run `main.ipynb` cells sequentially.

### Accessing Jupyter from a Remote Server

If running on a remote server, use one of these methods:

- **Direct access**: If the server has a public IP/hostname, open `http://<server-ip>:8888` in your browser
- **SSH tunnel** (recommended for security): On your local machine, run:
  ```bash
  ssh -L 8888:localhost:8888 <username>@<server-ip>
  ```
  Then open `http://localhost:8888` in your browser.

## Manual Docker Run

If you prefer not to use docker-compose:

```bash
docker run -d --gpus all -v $(pwd):/app -v $(pwd)/weights:/weights -v $(pwd)/comfyui_models:/comfyui_models -e PYTHONPATH=/app/swan --name swan-container swan:latest
docker exec -it swan-container bash
```

## Console Script

A console-based script (`main.py`) provides an interactive alternative to the Jupyter notebook with checkpoint support:

Inside the container (after `docker compose exec swan bash` or manual run):

```bash
# Run the full pipeline
python main.py --n-drones 500 --video-name trex --user-prompt "A T-Rex roaring and turning its head from left to right standing on a large flat rock."

# After each stage, you'll see the result path and be prompted to continue.
```

### Restarting from a Stage

If you stop the script mid-pipeline, you can resume by specifying the starting stage:

```bash
# Restart from tracking (requires video generation outputs)
python main.py --start-stage tracking --video-path /app/out/trex_500_drones/video_gen/video.mp4 --prompt-path /app/out/trex_500_drones/video_gen/prompt.json

# Restart from trajectory generation (requires tracking outputs)
python main.py --start-stage trajectory --tracking-dir /app/out/trex_500_drones/tracking

# Restart from simulation (requires trajectory outputs)
python main.py --start-stage simulation
```

Stage options: `video`, `segmentation`, `tracking`, `trajectory`, `simulation`, `visualization`
# SWAN - Swarm Animation From Natural Language
SWAN is generative AI pipeline that creates dynamic choreographies for drone light shows with thousands of UAVs, based on natural language descriptions.

## Setup

```bash
# Clone the repository with the --recursive flag to include the comfyui submodule
git clone --recursive https://github.com/SWAN-URL/repo-name.git
```

Models will be cached in `./weights` and `./comfyui_models` on your host filesystem.

## Installation

```bash
# Build the image
docker build -t swan .

# Download model weights (choose 1 for core models only, or 2 for all including video models)
docker compose run swan python download_weights.py
```

## Run SWAN

Run the gradio UI:
```bash
docker compose run swan python main.py
```

You then can find the results in ./out/\<name\>/.

## System Requirements
The system should have an Nvidia GPU, with at least 16 GB VRAM. Furthermore, we receommend at least 32 GB of RAM and 100 GB of free disk space.

## Entry points
For a detailed explanation of the codebase, please refer to [AGENTS.md](./AGENTS.md).

## Citation
```bibtex
@misc{reinhold2026generativeaisafephotorealistic,
      title={Generative AI for Safe and Photorealistic Drone Light Shows}, 
      author={Pascal Reinhold and Alexander Gräfe and Sebastian Trimpe},
      year={2026},
      url={https://arxiv.org/abs/2606.25458}, 
}
```

# SWAN - Swarm Animation From Natural Language
SWAN is generative AI pipeline that creates dynamic choreographies for drone light shows with thousands of UAVs, based on natural language descriptions.


# Setup

1. Clone the repository with the `--recursive` flag to include the submodule for comfyui.
2. Download the docker image from github packages
3. Run the docker container with the appropriate environment variables and volume mounts. (see docker-compose.yaml for reference)
4. Download the weights by running `python download_weights.py` inside the container. The script is interactive and allows to skip the weights for video generation if desired (~56GB). In this case only usage with custom videos is possible, example videos are provided (./examples/videos).

```
git clone --recursive https://github.com/SWAN-URL/repo-name.git
```
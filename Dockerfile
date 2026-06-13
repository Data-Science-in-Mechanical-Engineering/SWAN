FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

LABEL org.opencontainers.image.source="https://github.com/pascalreinhold/swan"

# environment variables
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    openssh-client \
    build-essential \
    ninja-build \
    git \
    curl \
    libgl1 libegl1 libglib2.0-0 \
    ffmpeg \
    rustc \
    cargo \
    ca-certificates


# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV --python 3.12
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


# Install base dependencies
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
uv pip install -r requirements.txt --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 --link-mode=copy

# install comfyui dependencies + flash attention to speed up video model inference
COPY comfyui/requirements.txt /app/comfyui/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
uv pip install -r comfyui/requirements.txt --link-mode=copy

RUN --mount=type=cache,target=/root/.cache/uv \
export MAX_JOBS=2 \
&& uv pip install flash-attn==2.8.3 --no-build-isolation
# ---------------------------------------------------------------------------------------------------------------------

CMD ["sleep", "infinity"]
# Runtime image for the retrieval baseline: CUDA 12.4 + Python 3.10 (managed by uv).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# uv binary from the official distroless image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PYTHON=3.10 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install a project-managed Python 3.10.
RUN uv python install 3.10

# Dependency layer: resolves and installs before copying sources for cache reuse.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project --no-dev

# Project layer.
COPY configs ./configs
COPY src ./src
COPY main.py ./
RUN uv sync --no-dev

# Data and index artifacts are expected to be mounted at runtime:
#   docker run --gpus all -v $PWD/data:/app/data -v $PWD/artifacts:/app/artifacts <image>
CMD ["uv", "run", "python", "main.py"]

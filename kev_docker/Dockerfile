ARG CUDA_VER=12.1.1
FROM nvidia/cuda:${CUDA_VER}-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ------------------------
# System + Python
# ------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    build-essential cmake ninja-build pkg-config \
    ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y --no-install-recommends \
  libegl1 libegl1-mesa libgles2 libgl1 \
  libglvnd0 libglx0 \
  mesa-utils \
  && rm -rf /var/lib/apt/lists/*
ENV PYOPENGL_PLATFORM=egl


# python / pip alias
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ------------------------
# Python tooling
# ------------------------
RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# ------------------------
# PyTorch (CUDA 12.1, RTX 4090 OK)
# ------------------------
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ------------------------
# Common Python deps
# ------------------------
RUN pip install --no-cache-dir \
    pytorch-lightning pyrender opencv-python yacs scikit-image einops timm \
    dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog \
    pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn \
    wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools \
    loguru optree fvcore black pycocotools tensorboard huggingface_hub

# ------------------------
# Detectron2 (pinned commit)
# ------------------------
RUN pip install --no-cache-dir \
    'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps

# ------------------------
# MoGe (Microsoft)
# ------------------------
RUN pip install --no-cache-dir \
    git+https://github.com/microsoft/MoGe.git

# ------------------------
# SAM3
# ------------------------
RUN cd /workspace && \
    git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir decord psutil

# ------------------------
# Default worker shell
# ------------------------
CMD ["/bin/bash"]


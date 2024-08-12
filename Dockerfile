FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# DUST3R
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/tauzn-clock/Depth-Anything-V2 /depthanything
WORKDIR /depthanything
RUN pip install -r requirements.txt
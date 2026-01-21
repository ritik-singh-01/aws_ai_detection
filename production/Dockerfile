# =============================================================================
# Dream Vision - Production Dockerfile
# Multi-stage build for lightweight, optimized container
# Target: AWS G6 (NVIDIA L4 GPU) / Any CUDA-capable GPU
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and download models
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 (specific versions for reproducibility)
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install dependencies
COPY requirements_cloud.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Pre-download the model to cache it in the image (optional - makes image larger but faster startup)
# Uncomment if you want model baked into image (~5GB larger)
# RUN python -c "from diffusers import AutoPipelineForImage2Image; AutoPipelineForImage2Image.from_pretrained('stabilityai/sdxl-turbo', torch_dtype='float16', variant='fp16')"

# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal runtime image
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS production

# Labels for container registry
LABEL maintainer="DreamVision Team"
LABEL version="1.0.0"
LABEL description="Dream Vision AI Art Generation - Production Image"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash dreamvision
USER dreamvision

# Create app directory
WORKDIR /app

# Copy application code
COPY --chown=dreamvision:dreamvision dream_vision_cloud.py .

# Create directories for input/output
RUN mkdir -p /app/input /app/output /app/models

# Set model cache directory
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV TORCH_HOME=/app/models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3.11 -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command
ENTRYPOINT ["python3.11", "dream_vision_cloud.py"]
CMD ["--help"]

# -----------------------------------------------------------------------------
# Usage Examples:
# -----------------------------------------------------------------------------
# Build:
#   docker build -t dreamvision:latest -f Dockerfile ..
#
# Run with GPU:
#   docker run --gpus all -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
#       dreamvision:latest --input /app/input/video.mp4 --output /app/output/result.mp4
#
# Interactive shell:
#   docker run --gpus all -it --entrypoint /bin/bash dreamvision:latest
# -----------------------------------------------------------------------------

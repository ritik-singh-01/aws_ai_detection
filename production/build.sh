#!/bin/bash
# =============================================================================
# Dream Vision - Production Build Script
# Quick build and run commands for Docker deployment
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
IMAGE_NAME="dreamvision"
VERSION=$(cat VERSION 2>/dev/null || echo "1.0.0")

echo ""
echo "============================================================"
echo -e "${GREEN}  Dream Vision - Production Build${NC}"
echo "  Version: ${VERSION}"
echo "============================================================"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[Error] Docker is not installed!${NC}"
    exit 1
fi

# Check NVIDIA Docker support
if ! docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}[Warning] NVIDIA GPU support not available${NC}"
    echo "Container will build but GPU features won't work"
fi

# Parse command
COMMAND=${1:-help}

case $COMMAND in
    build)
        echo -e "${GREEN}[1/2] Building Docker image...${NC}"
        cd ..
        docker build -t ${IMAGE_NAME}:${VERSION} -t ${IMAGE_NAME}:latest \
            -f production/Dockerfile .
        echo ""
        echo -e "${GREEN}[2/2] Build complete!${NC}"
        echo "Image: ${IMAGE_NAME}:${VERSION}"
        docker images ${IMAGE_NAME}
        ;;

    run)
        INPUT=${2:-"/app/input/video.mp4"}
        OUTPUT=${3:-"/app/output/result.mp4"}

        echo -e "${GREEN}Running Dream Vision...${NC}"
        echo "Input: ${INPUT}"
        echo "Output: ${OUTPUT}"
        echo ""

        # Create directories
        mkdir -p input output

        docker run --rm --gpus all \
            -v $(pwd)/input:/app/input \
            -v $(pwd)/output:/app/output \
            -v dreamvision-models:/app/models \
            ${IMAGE_NAME}:latest \
            --input "${INPUT}" \
            --output "${OUTPUT}"
        ;;

    batch)
        echo -e "${GREEN}Running batch processing...${NC}"
        mkdir -p input output
        docker-compose --profile batch up dreamvision-batch
        ;;

    shell)
        echo -e "${GREEN}Starting interactive shell...${NC}"
        docker run --rm --gpus all -it \
            -v $(pwd)/input:/app/input \
            -v $(pwd)/output:/app/output \
            -v dreamvision-models:/app/models \
            --entrypoint /bin/bash \
            ${IMAGE_NAME}:latest
        ;;

    test)
        echo -e "${GREEN}Testing Docker image...${NC}"
        docker run --rm ${IMAGE_NAME}:latest python3.11 -c "
import torch
import cv2
import numpy as np
from PIL import Image
print('All imports successful!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
        ;;

    gpu-test)
        echo -e "${GREEN}Testing GPU access...${NC}"
        docker run --rm --gpus all ${IMAGE_NAME}:latest python3.11 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print('GPU test passed!')
"
        ;;

    push)
        REGISTRY=${2:-""}
        if [ -z "$REGISTRY" ]; then
            echo -e "${RED}[Error] Registry URL required${NC}"
            echo "Usage: ./build.sh push <registry-url>"
            exit 1
        fi

        echo -e "${GREEN}Pushing to ${REGISTRY}...${NC}"
        docker tag ${IMAGE_NAME}:latest ${REGISTRY}/${IMAGE_NAME}:latest
        docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:${VERSION}
        docker push ${REGISTRY}/${IMAGE_NAME}:latest
        docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
        echo "Push complete!"
        ;;

    clean)
        echo -e "${YELLOW}Cleaning up...${NC}"
        docker rmi ${IMAGE_NAME}:latest ${IMAGE_NAME}:${VERSION} 2>/dev/null || true
        docker image prune -f
        echo "Cleanup complete!"
        ;;

    help|*)
        echo "Usage: ./build.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  build              Build Docker image"
        echo "  run [in] [out]     Run with input/output paths"
        echo "  batch              Process all videos in input/"
        echo "  shell              Start interactive shell"
        echo "  test               Test image (no GPU)"
        echo "  gpu-test           Test GPU access"
        echo "  push <registry>    Push to registry"
        echo "  clean              Remove images"
        echo "  help               Show this help"
        echo ""
        echo "Examples:"
        echo "  ./build.sh build"
        echo "  ./build.sh run /app/input/video.mp4 /app/output/result.mp4"
        echo "  ./build.sh push 123456789.dkr.ecr.us-east-1.amazonaws.com"
        ;;
esac

echo ""

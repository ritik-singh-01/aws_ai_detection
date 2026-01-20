#!/bin/bash
################################################################################
# Dream Vision - AWS G6 (L4 GPU) Complete Setup Script
#
# Target Instance: g6.xlarge / g6.2xlarge / g6.4xlarge
# GPU: NVIDIA L4 (24GB VRAM)
# AMI: Ubuntu 22.04 LTS with NVIDIA drivers
#
# Usage:
#   chmod +x setup_aws.sh
#   ./setup_aws.sh
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "============================================================"
echo -e "${GREEN}  Dream Vision - AWS G6 (L4 GPU) Setup${NC}"
echo "  NVIDIA L4 GPU - 24GB VRAM"
echo "============================================================"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}[Warning] Running as root. Consider using a regular user.${NC}"
fi

# ============================================================
# STEP 1: Update System
# ============================================================
echo -e "${GREEN}[1/9] Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# ============================================================
# STEP 2: Install Python 3.11
# ============================================================
echo -e "${GREEN}[2/9] Installing Python 3.11...${NC}"
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Verify Python
python3.11 --version

# ============================================================
# STEP 3: Check/Install NVIDIA Drivers
# ============================================================
echo -e "${GREEN}[3/9] Checking NVIDIA drivers...${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers found:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo -e "${YELLOW}Installing NVIDIA drivers...${NC}"
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall

    echo -e "${YELLOW}============================================================${NC}"
    echo -e "${YELLOW}  NVIDIA drivers installed. REBOOT REQUIRED!${NC}"
    echo -e "${YELLOW}  Run: sudo reboot${NC}"
    echo -e "${YELLOW}  Then run this script again after reboot.${NC}"
    echo -e "${YELLOW}============================================================${NC}"
    exit 0
fi

# ============================================================
# STEP 4: Install System Dependencies
# ============================================================
echo -e "${GREEN}[4/9] Installing system dependencies...${NC}"
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    htop \
    nvtop

# ============================================================
# STEP 5: Create Project Directory
# ============================================================
echo -e "${GREEN}[5/9] Creating project directory...${NC}"
PROJECT_DIR=~/dream_vision
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# ============================================================
# STEP 6: Create Virtual Environment
# ============================================================
echo -e "${GREEN}[6/9] Creating Python virtual environment...${NC}"
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# ============================================================
# STEP 7: Install PyTorch with CUDA
# ============================================================
echo -e "${GREEN}[7/9] Installing PyTorch with CUDA 12.1...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ============================================================
# STEP 8: Install Project Dependencies
# ============================================================
echo -e "${GREEN}[8/9] Installing project dependencies...${NC}"

# xformers for memory optimization
pip install xformers

# Diffusers and Transformers
pip install diffusers transformers accelerate

# OpenCV
pip install opencv-python opencv-contrib-python

# MediaPipe (for person segmentation)
pip install mediapipe==0.10.9

# Other dependencies
pip install Pillow numpy tqdm safetensors

# ============================================================
# STEP 9: Verify Installation
# ============================================================
echo -e "${GREEN}[9/9] Verifying installation...${NC}"
echo ""

python -c "
import torch
print('=' * 60)
print('CUDA Verification')
print('=' * 60)
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'GPU Memory: {props.total_memory / 1024**3:.1f} GB')
    print(f'Compute Capability: {props.major}.{props.minor}')
print('=' * 60)
"

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Project directory: $PROJECT_DIR"
echo ""
echo "Next steps:"
echo "  1. Upload your video file to: $PROJECT_DIR/input.mp4"
echo "  2. Upload dream_vision_cloud.py to: $PROJECT_DIR/"
echo "  3. Activate environment: source venv/bin/activate"
echo "  4. Run: python dream_vision_cloud.py --input input.mp4"
echo ""
echo "Quick commands:"
echo "  cd $PROJECT_DIR && source venv/bin/activate"
echo ""

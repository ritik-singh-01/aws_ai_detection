# Dream Vision - Complete Deployment Guide

Comprehensive guide for deploying Dream Vision locally, on AWS, or using Docker containers.

---

## Table of Contents

1. [Deployment Options](#1-deployment-options)
2. [Local Deployment (Recommended)](#2-local-deployment-recommended)
3. [Docker Deployment](#3-docker-deployment)
4. [AWS Cloud Deployment](#4-aws-cloud-deployment)
5. [Jenkins CI/CD Setup](#5-jenkins-cicd-setup)
6. [Monitoring & Logging](#6-monitoring--logging)
7. [Security Best Practices](#7-security-best-practices)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Deployment Options

| Option | Best For | Hardware | Setup Time |
|--------|----------|----------|------------|
| **Local Webcam** | Personal use, demos | Your laptop/PC with GPU | Quick |
| **Docker Local** | Isolated environment | Your machine with GPU | Medium |
| **AWS Cloud** | High resolution, batch processing | AWS G6 instances | Medium |
| **Kubernetes** | Enterprise scale | Cloud infrastructure | Complex |

### Quick Decision Guide

- **Just want to try it?** -> Local Webcam
- **Need isolation?** -> Docker Local
- **Need more GPU power?** -> AWS Cloud
- **Enterprise deployment?** -> Docker + Kubernetes

---

## 2. Local Deployment (Recommended)

### 2.1 Prerequisites

| Component | Required | Check Command |
|-----------|----------|---------------|
| **Python** | 3.10 or 3.11 | `python --version` |
| **NVIDIA GPU** | 8GB+ VRAM | `nvidia-smi` |
| **CUDA** | 12.x | `nvidia-smi` |
| **OS** | Windows 10/11 or Linux | - |

> **WARNING**: Python 3.14 is NOT supported. Use Python 3.10 or 3.11.

### 2.2 Installation (Windows)

#### Option A: Automated Install
```batch
install.bat
```

#### Option B: Manual Install
```bash
# 1. Create virtual environment (Python 3.10 or 3.11)
python -m venv venv

# 2. Activate
venv\Scripts\activate

# 3. Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install xformers (CRITICAL for 8GB VRAM)
pip install xformers

# 5. Install AI libraries
pip install diffusers transformers accelerate

# 6. Install computer vision (MediaPipe 0.10.9 required)
pip install opencv-python mediapipe==0.10.9

# 7. Install utilities
pip install Pillow numpy tqdm
```

### 2.3 Installation (Linux)

```bash
# 1. Install Python 3.11
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install xformers
pip install xformers

# 5. Install all dependencies
pip install diffusers transformers accelerate opencv-python mediapipe==0.10.9 Pillow numpy tqdm
```

### 2.4 Verify Installation

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 3070 Laptop GPU
```

### 2.5 Run with Webcam

```bash
# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux

# Run cloud version locally (recommended - has all features)
python aws_cloud_deployment/dream_vision_cloud.py --mode webcam

# Or run simpler versions
python dream_vision.py
python dream_vision_optimized.py
```

### 2.6 Run Options

```bash
# Basic webcam mode
python aws_cloud_deployment/dream_vision_cloud.py --mode webcam

# Custom prompt
python aws_cloud_deployment/dream_vision_cloud.py --mode webcam \
    --prompt "ethereal oil painting, renaissance style, golden light"

# Adjust transformation strength (0.0-1.0)
python aws_cloud_deployment/dream_vision_cloud.py --mode webcam --strength 0.4

# For 8GB VRAM (lower resolution)
python aws_cloud_deployment/dream_vision_cloud.py --mode webcam --width 512 --height 512

# Adjust face preservation (default 0.85)
python aws_cloud_deployment/dream_vision_cloud.py --mode webcam --face-preserve 0.9

# Disable gender detection
python aws_cloud_deployment/dream_vision_cloud.py --mode webcam --no-gender-detection
```

### 2.7 Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit application |
| `F` | Toggle Fullscreen |
| `S` | Save Screenshot |

### 2.8 Performance Settings by GPU

| GPU | VRAM | Resolution | Steps | Expected FPS |
|-----|------|------------|-------|--------------|
| RTX 3060 | 8 GB | 512x512 | 1 | 10-12 |
| RTX 3070 | 8 GB | 512x512 | 1 | 12-15 |
| RTX 3080 | 10 GB | 768x768 | 1 | 15-20 |
| RTX 4070 | 12 GB | 768x768 | 2 | 15-18 |
| RTX 4080 | 16 GB | 768x768 | 2 | 20-25 |

---

## 3. Docker Deployment

### 3.1 Prerequisites

```bash
# Docker with NVIDIA support required
Docker >= 24.0
NVIDIA Container Toolkit
Docker Compose >= 2.20
```

### 3.2 Install NVIDIA Container Toolkit (Linux)

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### 3.3 Build Docker Image

```bash
# Navigate to project directory
cd aws_cloud_deployment

# Build the image
docker build -t dreamvision:latest -f production/Dockerfile .

# Verify image
docker images dreamvision
```

### 3.4 Run Container (Video Processing)

```bash
# Create directories
mkdir -p input output

# Copy your video
cp /path/to/video.mp4 input/

# Run with GPU
docker run --rm --gpus all \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    dreamvision:latest \
    --input /app/input/video.mp4 \
    --output /app/output/result.mp4
```

### 3.5 Docker Compose Usage

```bash
# Navigate to production folder
cd production

# Create directories
mkdir -p input output logs

# Build
docker-compose build

# Process single video
docker-compose run --rm dreamvision \
    --input /app/input/video.mp4 \
    --output /app/output/result.mp4

# Batch process all videos
docker-compose --profile batch up dreamvision-batch

# Stop all
docker-compose down
```

### 3.6 Docker Image Optimization

**Applied optimizations:**

| Optimization | Effect |
|--------------|--------|
| Multi-stage build | Separates build and runtime |
| Minimal base image | Uses CUDA runtime (not devel) |
| Non-root user | Security best practice |
| Layer caching | Optimized COPY order |
| .dockerignore | Excludes unnecessary files |
| No pip cache | PIP_NO_CACHE_DIR=1 |
| Volume for models | Models cached externally |

**Image size:**

| Layer | Size |
|-------|------|
| Base CUDA Runtime | ~3.5 GB |
| Python + Dependencies | ~2.5 GB |
| Application | ~50 KB |
| **Total Image** | **~6 GB** |

---

## 4. AWS Cloud Deployment

### 4.1 Instance Selection

| Instance | vCPUs | RAM | GPU | VRAM | Cost/hr* |
|----------|-------|-----|-----|------|----------|
| g6.xlarge | 4 | 16 GB | 1x L4 | 24 GB | ~$0.80 |
| g6.2xlarge | 8 | 32 GB | 1x L4 | 24 GB | ~$1.00 |
| g5.xlarge | 4 | 16 GB | 1x A10G | 24 GB | ~$1.00 |

*Use Spot Instances for 60-90% savings!

### 4.2 Launch EC2 Instance

1. **Go to AWS Console** -> EC2 -> Launch Instance
2. **Choose AMI**: Deep Learning AMI GPU PyTorch 2.1 (Ubuntu 22.04)
3. **Instance Type**: g6.xlarge or g6.2xlarge
4. **Key Pair**: Create new or use existing
5. **Security Group**: Allow SSH (port 22) from your IP
6. **Storage**: 100 GB gp3 SSD
7. **Launch Instance**

### 4.3 Connect and Setup

```bash
# Connect via SSH
ssh -i dreamvision-key.pem ubuntu@<INSTANCE_IP>

# Upload project files (from local machine)
scp -i dreamvision-key.pem -r aws_cloud_deployment/* ubuntu@<IP>:~/dream_vision/

# On EC2: Run setup
cd ~/dream_vision
chmod +x setup_aws.sh
./setup_aws.sh
```

### 4.4 Process Videos on AWS

```bash
# Activate environment
source venv/bin/activate

# Process video (768x768 default for L4)
python dream_vision_cloud.py --input input.mp4 --output output.mp4

# High resolution (L4 can handle 1024x1024)
python dream_vision_cloud.py --input input.mp4 --output output_hd.mp4 --width 1024 --height 1024
```

### 4.5 Download Results

```bash
# From local machine
scp -i dreamvision-key.pem ubuntu@<IP>:~/dream_vision/output.mp4 ./
```

### 4.6 Cost Optimization

| Tip | Savings |
|-----|---------|
| Use Spot Instances | 60-90% |
| Stop instance when not using | 100% compute |
| Use t3.medium for setup, then switch to G6 | ~50% |

---

## 5. Jenkins CI/CD Setup

### 5.1 Jenkins Requirements

```
Jenkins Plugins:
- Docker Pipeline
- Pipeline: AWS Steps
- Git Plugin
- Credentials Binding
```

### 5.2 Configure Credentials

```groovy
// Add in Jenkins Credentials:

// 1. Docker Registry
ID: docker-registry-credentials
Type: Username/Password

// 2. AWS Credentials
ID: aws-credentials
Type: AWS Credentials

// 3. Docker Registry URL
ID: docker-registry-url
Type: Secret text
```

### 5.3 Pipeline Stages

| Stage | Description |
|-------|-------------|
| Checkout | Clone repository |
| Code Quality | Lint & security scan |
| Build Image | Docker build |
| Test Image | Verify dependencies |
| GPU Test | Test CUDA (if available) |
| Push Registry | Push to ECR/DockerHub |
| Deploy | Update service |

### 5.4 Push to Amazon ECR

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    123456789.dkr.ecr.us-east-1.amazonaws.com

# Create repository
aws ecr create-repository --repository-name dreamvision

# Tag and push
docker tag dreamvision:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/dreamvision:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/dreamvision:latest
```

---

## 6. Monitoring & Logging

### 6.1 GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Or use nvtop (if installed)
nvtop
```

### 6.2 Docker Logs

```bash
# View container logs
docker logs -f dreamvision-processor

# With timestamps
docker logs -t dreamvision-processor
```

### 6.3 Processing Progress

The application prints progress every 2 seconds:
```
[Progress] 45.2% | FPS: 28.3 | Latency: 35ms | Frames: 542 | Elapsed: 19s
```

---

## 7. Security Best Practices

### 7.1 Container Security

```
Implemented:
- Non-root user (dreamvision:1000)
- Read-only root filesystem compatible
- No unnecessary packages
- Multi-stage build (no build tools in runtime)
```

### 7.2 Network Security

```bash
# AWS Security Group recommendations:
- Allow SSH (22) from your IP only
- No other inbound ports needed for video processing
```

### 7.3 Secrets Management

```bash
# Use AWS Secrets Manager for tokens
aws secretsmanager create-secret \
    --name dreamvision/hf-token \
    --secret-string "hf_xxxxx"
```

---

## 8. Troubleshooting

### Python Version Error
```
ERROR: PyTorch CUDA requires Python 3.10 or 3.11
```
**Solution**: Install Python 3.11 from python.org

### CUDA Not Available
```bash
# Check NVIDIA driver
nvidia-smi

# If not found (Linux):
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Out of Memory (OOM)
```bash
# Reduce resolution
--width 512 --height 512

# Reduce strength
--strength 0.35
```

### MediaPipe Import Error
```bash
pip uninstall mediapipe
pip install mediapipe==0.10.9
```

### Camera Not Found
```bash
# Try different camera IDs
--camera 1  # or 2, 3...
```

### Docker GPU Not Available
```bash
# Check NVIDIA Container Toolkit
nvidia-container-cli info

# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Permission Denied on Volumes
```bash
# Fix ownership
sudo chown -R 1000:1000 input output models
```

### Connection Lost During Cloud Processing
```bash
# Use screen to keep process running
screen -S dreamvision
python dream_vision_cloud.py --input input.mp4

# Detach: Ctrl+A, then D
# Reattach: screen -r dreamvision
```

---

## Quick Reference Commands

### Local Webcam
```bash
# Activate and run
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux
python aws_cloud_deployment/dream_vision_cloud.py --mode webcam
```

### Docker
```bash
# Build
docker-compose build

# Run video processing
docker-compose run --rm dreamvision \
    --input /app/input/video.mp4 \
    --output /app/output/result.mp4
```

### AWS
```bash
# Connect
ssh -i key.pem ubuntu@<IP>

# Upload
scp -i key.pem video.mp4 ubuntu@<IP>:~/dream_vision/

# Process
python dream_vision_cloud.py --input video.mp4 --output result.mp4

# Download
scp -i key.pem ubuntu@<IP>:~/dream_vision/result.mp4 ./
```

---

## Project Files

```
DreamVision/
|-- dream_vision.py              # Standard version
|-- dream_vision_optimized.py    # Optimized version
|-- requirements.txt             # Dependencies
|-- install.bat                  # Windows installer
|-- run.bat                      # Windows launcher
|-- README.md                    # Main documentation
|
|-- aws_cloud_deployment/
    |-- dream_vision_cloud.py    # Full-featured version
    |-- setup_aws.sh             # AWS setup script
    |-- requirements_cloud.txt   # Cloud dependencies
    |-- run_cloud.sh             # Cloud runner
    |-- README.md                # AWS guide
    |
    |-- production/
        |-- Dockerfile           # Docker image build
        |-- docker-compose.yml   # Container orchestration
        |-- Jenkinsfile          # CI/CD pipeline
        |-- build.sh             # Build commands
        |-- PRODUCTION_GUIDE.md  # This file
```

---

**Ready to deploy!**

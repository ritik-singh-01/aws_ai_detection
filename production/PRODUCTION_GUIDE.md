# Dream Vision - Production Deployment Guide

Complete guide for deploying Dream Vision in production using Docker, Jenkins, and AWS.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Docker Setup](#3-docker-setup)
4. [Jenkins CI/CD Setup](#4-jenkins-cicd-setup)
5. [AWS Deployment](#5-aws-deployment)
6. [Kubernetes Deployment](#6-kubernetes-deployment-optional)
7. [Monitoring & Logging](#7-monitoring--logging)
8. [Security Best Practices](#8-security-best-practices)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Production Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────┐     ┌──────────┐     ┌──────────────────────────┐   │
│   │  GitHub  │────▶│ Jenkins  │────▶│  Docker Registry (ECR)   │   │
│   │   Repo   │     │ Pipeline │     │  dreamvision:latest      │   │
│   └──────────┘     └──────────┘     └──────────────────────────┘   │
│                                                │                     │
│                                                ▼                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │                    AWS G6 Instance                        │      │
│   │  ┌────────────────────────────────────────────────────┐  │      │
│   │  │              Docker Container                       │  │      │
│   │  │  ┌──────────────────────────────────────────────┐  │  │      │
│   │  │  │         Dream Vision Application             │  │  │      │
│   │  │  │  • dream_vision_cloud.py                     │  │  │      │
│   │  │  │  • PyTorch + CUDA                            │  │  │      │
│   │  │  │  • SDXL-Turbo Model                          │  │  │      │
│   │  │  └──────────────────────────────────────────────┘  │  │      │
│   │  │                      │                              │  │      │
│   │  │              ┌───────▼───────┐                      │  │      │
│   │  │              │  NVIDIA L4    │                      │  │      │
│   │  │              │   24GB VRAM   │                      │  │      │
│   │  │              └───────────────┘                      │  │      │
│   │  └────────────────────────────────────────────────────┘  │      │
│   │                                                           │      │
│   │  Volumes:                                                 │      │
│   │  • /input  (video files)                                  │      │
│   │  • /output (processed files)                              │      │
│   │  • /models (cached AI models)                             │      │
│   └───────────────────────────────────────────────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Image Size Optimization

| Layer | Size | Description |
|-------|------|-------------|
| Base CUDA Runtime | ~3.5 GB | nvidia/cuda:12.1.1-cudnn8-runtime |
| Python + Deps | ~2.5 GB | PyTorch, diffusers, opencv |
| Application | ~50 KB | dream_vision_cloud.py |
| **Total Image** | **~6 GB** | Without model cache |

**Note:** Model (~5GB) is downloaded at first run and cached in volume.

---

## 2. Prerequisites

### Local Development Machine

```bash
# Docker with NVIDIA support
Docker >= 24.0
NVIDIA Container Toolkit
Docker Compose >= 2.20

# Optional for CI/CD
Jenkins >= 2.400
AWS CLI >= 2.0
```

### AWS Requirements

```
- AWS Account with EC2 access
- IAM permissions for ECR, EC2, ECS
- g6.xlarge or g6.2xlarge quota
- VPC with GPU instance support
```

### Install NVIDIA Container Toolkit (Linux)

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

---

## 3. Docker Setup

### 3.1 Build Docker Image

```bash
# Navigate to project directory
cd aws_cloud_deployment

# Build the image
docker build -t dreamvision:latest -f production/Dockerfile .

# Verify image
docker images dreamvision
```

### 3.2 Run Container (Single Video)

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

### 3.3 Docker Compose Usage

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

### 3.4 Docker Image Optimization Tips

**Current optimizations applied:**

1. **Multi-stage build** - Separates build and runtime dependencies
2. **Minimal base image** - Uses CUDA runtime (not devel)
3. **Non-root user** - Security best practice
4. **Layer caching** - Optimized COPY order
5. **.dockerignore** - Excludes unnecessary files
6. **No pip cache** - `PIP_NO_CACHE_DIR=1`
7. **Volume for models** - Models cached externally

**Additional optimization (if needed):**

```dockerfile
# Slim Python packages (add to Dockerfile)
RUN pip install --no-deps <package>  # Install without dependencies

# Remove unnecessary files
RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + && \
    find /opt/venv -type f -name "*.pyc" -delete
```

---

## 4. Jenkins CI/CD Setup

### 4.1 Jenkins Requirements

```
Jenkins Plugins:
- Docker Pipeline
- Pipeline: AWS Steps
- Git Plugin
- Credentials Binding
- Slack Notification (optional)
```

### 4.2 Configure Jenkins Credentials

```groovy
// Add these credentials in Jenkins:

// 1. Docker Registry
ID: docker-registry-credentials
Type: Username/Password
Username: your-docker-username
Password: your-docker-token

// 2. AWS Credentials
ID: aws-credentials
Type: AWS Credentials
Access Key ID: AKIA...
Secret Access Key: ...

// 3. Docker Registry URL
ID: docker-registry-url
Type: Secret text
Value: your-registry.com
```

### 4.3 Create Jenkins Pipeline

```bash
# Option 1: Pipeline from SCM
1. New Item > Pipeline
2. Pipeline > Definition: Pipeline script from SCM
3. SCM: Git
4. Repository URL: https://github.com/your/repo.git
5. Script Path: aws_cloud_deployment/production/Jenkinsfile

# Option 2: Copy Jenkinsfile content directly
1. New Item > Pipeline
2. Pipeline > Definition: Pipeline script
3. Paste Jenkinsfile content
```

### 4.4 Configure Build Agent

For GPU testing, create a build agent with:

```yaml
# Jenkins agent with GPU (Docker)
docker run -d \
    --name jenkins-agent \
    --gpus all \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -e JENKINS_URL=http://jenkins:8080 \
    -e JENKINS_AGENT_NAME=docker-gpu \
    jenkins/inbound-agent
```

### 4.5 Pipeline Stages

| Stage | Description | Duration |
|-------|-------------|----------|
| Checkout | Clone repository | ~10s |
| Code Quality | Lint & security scan | ~30s |
| Build Image | Multi-stage Docker build | ~5-10min |
| Test Image | Verify dependencies | ~30s |
| GPU Test | Test CUDA (if available) | ~30s |
| Push Registry | Push to ECR/DockerHub | ~2-5min |
| Deploy | Update ECS/EC2 service | ~1-2min |

---

## 5. AWS Deployment

### 5.1 Push to Amazon ECR

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    123456789.dkr.ecr.us-east-1.amazonaws.com

# Create repository (first time)
aws ecr create-repository --repository-name dreamvision

# Tag and push
docker tag dreamvision:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/dreamvision:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/dreamvision:latest
```

### 5.2 Launch EC2 G6 Instance

```bash
# Using AWS CLI
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --instance-type g6.xlarge \
    --key-name your-key \
    --security-group-ids sg-xxxxx \
    --subnet-id subnet-xxxxx \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=DreamVision-Production}]'
```

### 5.3 Deploy on EC2

```bash
# SSH to instance
ssh -i key.pem ubuntu@<instance-ip>

# Install Docker and NVIDIA Container Toolkit
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Login to ECR
aws configure  # Enter your credentials
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Pull and run
docker pull 123456789.dkr.ecr.us-east-1.amazonaws.com/dreamvision:latest

# Create directories
mkdir -p ~/dreamvision/{input,output,models}

# Run
docker run --rm --gpus all \
    -v ~/dreamvision/input:/app/input \
    -v ~/dreamvision/output:/app/output \
    -v ~/dreamvision/models:/app/models \
    123456789.dkr.ecr.us-east-1.amazonaws.com/dreamvision:latest \
    --input /app/input/video.mp4 \
    --output /app/output/result.mp4
```

### 5.4 Auto-Scaling (Optional)

```bash
# Create launch template with user data
#!/bin/bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker pull 123456789.dkr.ecr.us-east-1.amazonaws.com/dreamvision:latest

# Create Auto Scaling Group
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name dreamvision-asg \
    --launch-template LaunchTemplateId=lt-xxx \
    --min-size 0 \
    --max-size 5 \
    --desired-capacity 1
```

---

## 6. Kubernetes Deployment (Optional)

### 6.1 Kubernetes Manifest

```yaml
# dreamvision-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dreamvision
  labels:
    app: dreamvision
spec:
  replicas: 1
  selector:
    matchLabels:
      app: dreamvision
  template:
    metadata:
      labels:
        app: dreamvision
    spec:
      containers:
      - name: dreamvision
        image: 123456789.dkr.ecr.us-east-1.amazonaws.com/dreamvision:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
        volumeMounts:
        - name: input
          mountPath: /app/input
        - name: output
          mountPath: /app/output
        - name: models
          mountPath: /app/models
      volumes:
      - name: input
        persistentVolumeClaim:
          claimName: dreamvision-input
      - name: output
        persistentVolumeClaim:
          claimName: dreamvision-output
      - name: models
        persistentVolumeClaim:
          claimName: dreamvision-models
      nodeSelector:
        nvidia.com/gpu: "true"
```

### 6.2 Deploy to EKS

```bash
# Apply manifest
kubectl apply -f dreamvision-deployment.yaml

# Check status
kubectl get pods -l app=dreamvision

# View logs
kubectl logs -f deployment/dreamvision
```

---

## 7. Monitoring & Logging

### 7.1 Docker Logs

```bash
# View container logs
docker logs -f dreamvision-processor

# With timestamps
docker logs -t dreamvision-processor
```

### 7.2 GPU Monitoring

```bash
# Install nvidia-exporter (docker-compose monitoring profile)
docker-compose --profile monitoring up -d

# Access metrics at http://localhost:9835/metrics

# Manual monitoring
watch -n 1 nvidia-smi
```

### 7.3 CloudWatch Integration

```bash
# Install CloudWatch agent on EC2
sudo yum install -y amazon-cloudwatch-agent

# Configure to collect GPU metrics
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
  "metrics": {
    "namespace": "DreamVision",
    "metrics_collected": {
      "nvidia_gpu": {
        "measurement": ["utilization_gpu", "memory_used"]
      }
    }
  }
}
EOF

# Start agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a start
```

---

## 8. Security Best Practices

### 8.1 Container Security

```dockerfile
# Already implemented in Dockerfile:
✅ Non-root user (dreamvision:1000)
✅ Read-only root filesystem compatible
✅ No unnecessary packages
✅ Multi-stage build (no build tools in runtime)
```

### 8.2 Network Security

```bash
# Restrict container network
docker run --network=none ...  # No network access
docker run --network=host ...  # Host network (for performance)

# AWS Security Group
- Allow SSH (22) from your IP only
- No inbound ports needed for processing
```

### 8.3 Secrets Management

```bash
# Use AWS Secrets Manager for HF_TOKEN
aws secretsmanager create-secret \
    --name dreamvision/hf-token \
    --secret-string "hf_xxxxx"

# Retrieve in container
export HF_TOKEN=$(aws secretsmanager get-secret-value \
    --secret-id dreamvision/hf-token \
    --query SecretString --output text)
```

---

## 9. Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Reduce resolution
docker run ... dreamvision --width 512 --height 512

# Check memory usage
nvidia-smi
```

**2. Model download fails**
```bash
# Pre-download model
docker run --rm -v dreamvision-models:/app/models dreamvision:latest \
    python -c "from diffusers import AutoPipelineForImage2Image; \
    AutoPipelineForImage2Image.from_pretrained('stabilityai/sdxl-turbo')"
```

**3. Docker GPU not available**
```bash
# Check NVIDIA Container Toolkit
nvidia-container-cli info

# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

**4. Permission denied on volumes**
```bash
# Fix ownership
sudo chown -R 1000:1000 input output models

# Or run as root (not recommended)
docker run --user root ...
```

---

## Quick Reference

```bash
# Build
docker-compose build

# Run single video
docker-compose run --rm dreamvision \
    --input /app/input/video.mp4 \
    --output /app/output/result.mp4

# Batch process
docker-compose --profile batch up dreamvision-batch

# View logs
docker-compose logs -f

# Stop all
docker-compose down

# Clean up
docker system prune -a
```

---

## Files in This Package

```
production/
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # Container orchestration
├── Jenkinsfile         # CI/CD pipeline
├── .dockerignore       # Build exclusions
└── PRODUCTION_GUIDE.md # This file
```

---

**Ready for production deployment!**

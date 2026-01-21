# Dream Vision - AWS Cloud Deployment Guide

Complete guide for deploying Dream Vision on AWS G6 instances with NVIDIA L4 GPU.

---

## Table of Contents

1. [AWS Instance Selection](#1-aws-instance-selection)
2. [Launch EC2 Instance](#2-launch-ec2-instance)
3. [Connect to Instance](#3-connect-to-instance)
4. [Upload Project Files](#4-upload-project-files)
5. [Run Setup Script](#5-run-setup-script)
6. [Process Videos](#6-process-videos)
7. [Download Results](#7-download-results)
8. [Cost Optimization](#8-cost-optimization)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. AWS Instance Selection

### Recommended: G6 Instances (NVIDIA L4)

| Instance Type | vCPUs | RAM | GPU | VRAM | Cost/hr* | Recommendation |
|---------------|-------|-----|-----|------|----------|----------------|
| g6.xlarge | 4 | 16 GB | 1x L4 | 24 GB | ~$0.80 | Best Value |
| g6.2xlarge | 8 | 32 GB | 1x L4 | 24 GB | ~$1.00 | Recommended |
| g6.4xlarge | 16 | 64 GB | 1x L4 | 24 GB | ~$1.50 | High Performance |
| g6.8xlarge | 32 | 128 GB | 1x L4 | 24 GB | ~$2.50 | Maximum CPU |

*Prices are approximate and vary by region. Use Spot Instances for 60-90% savings!

### Why L4 GPU?

- **24GB VRAM** - 3x more than RTX 3070 (8GB)
- **Ada Lovelace Architecture** - Latest generation
- **Higher Resolution** - Process at 768x768 or 1024x1024
- **Better FPS** - 25-40 FPS expected

### Alternative: G5 Instances (NVIDIA A10G)

| Instance Type | vCPUs | RAM | GPU | VRAM | Cost/hr* |
|---------------|-------|-----|-----|------|----------|
| g5.xlarge | 4 | 16 GB | 1x A10G | 24 GB | ~$1.00 |
| g5.2xlarge | 8 | 32 GB | 1x A10G | 24 GB | ~$1.20 |

---

## 2. Launch EC2 Instance

### Step-by-Step Instructions

#### A. Go to AWS Console
1. Login to [AWS Console](https://console.aws.amazon.com)
2. Navigate to **EC2** > **Instances** > **Launch Instance**

#### B. Choose AMI (Amazon Machine Image)
Select one of these AMIs:

**Option 1: Deep Learning AMI (Recommended)**
- Search: "Deep Learning AMI GPU PyTorch"
- Select: **Deep Learning AMI GPU PyTorch 2.1 (Ubuntu 22.04)**
- Pre-installed: NVIDIA drivers, CUDA, PyTorch

**Option 2: Ubuntu 22.04 LTS**
- Search: "Ubuntu 22.04"
- Select: **Ubuntu Server 22.04 LTS (HVM)**
- Requires: Manual NVIDIA driver installation

#### C. Choose Instance Type
- Select: **g6.xlarge** or **g6.2xlarge**
- Click "Compare instance types" to see specs

#### D. Configure Key Pair
1. Click **Create new key pair**
2. Name: `dreamvision-key`
3. Type: RSA
4. Format: .pem (Linux/Mac) or .ppk (Windows/PuTTY)
5. Download and save securely

#### E. Configure Network Settings
1. Allow SSH (port 22) from your IP
2. Create security group or use existing

#### F. Configure Storage
- Size: **100 GB** minimum (for model downloads)
- Type: **gp3** (SSD)
- Delete on termination: Yes (unless you want to keep data)

#### G. Advanced: Spot Instance (Optional - 60-90% savings)
1. Click **Advanced details**
2. Purchasing option: **Spot Instances**
3. Request type: **One-time**
4. Set maximum price (optional)

#### H. Launch Instance
1. Review settings
2. Click **Launch Instance**
3. Wait for instance to start (2-5 minutes)

---

## 3. Connect to Instance

### Get Instance IP
1. Go to **EC2** > **Instances**
2. Select your instance
3. Copy **Public IPv4 address**

### Connect via SSH

#### Linux/Mac Terminal:
```bash
# Set key permissions
chmod 400 dreamvision-key.pem

# Connect
ssh -i dreamvision-key.pem ubuntu@<YOUR_INSTANCE_IP>
```

#### Windows (PowerShell):
```powershell
ssh -i dreamvision-key.pem ubuntu@<YOUR_INSTANCE_IP>
```

#### Windows (PuTTY):
1. Open PuTTY
2. Host: `ubuntu@<YOUR_INSTANCE_IP>`
3. Connection > SSH > Auth > Credentials
4. Browse to your .ppk key file
5. Click Open

---

## 4. Upload Project Files

### Method 1: SCP (Recommended)

#### From Windows (PowerShell):
```powershell
# Upload entire aws_cloud_deployment folder
scp -i dreamvision-key.pem -r "C:\Ai project\DreamVision\aws_cloud_deployment\*" ubuntu@<IP>:~/dream_vision/

# Upload a video file
scp -i dreamvision-key.pem "C:\path\to\video.mp4" ubuntu@<IP>:~/dream_vision/input.mp4
```

#### From Linux/Mac:
```bash
# Upload project files
scp -i dreamvision-key.pem -r ./aws_cloud_deployment/* ubuntu@<IP>:~/dream_vision/

# Upload video
scp -i dreamvision-key.pem ./video.mp4 ubuntu@<IP>:~/dream_vision/input.mp4
```

### Method 2: Git Clone (if using repository)
```bash
# On the EC2 instance
cd ~
git clone <your-repo-url> dream_vision
cd dream_vision
```

### Method 3: AWS S3 (for large files)
```bash
# On your local machine - upload to S3
aws s3 cp video.mp4 s3://your-bucket/dream_vision/

# On EC2 instance - download from S3
aws s3 cp s3://your-bucket/dream_vision/video.mp4 ~/dream_vision/input.mp4
```

---

## 5. Run Setup Script

### On the EC2 Instance:

```bash
# Navigate to project directory
cd ~/dream_vision

# Make setup script executable
chmod +x setup_aws.sh

# Run setup script
./setup_aws.sh
```

### What the script does:
1. Updates system packages
2. Installs Python 3.11
3. Checks/installs NVIDIA drivers
4. Creates virtual environment
5. Installs PyTorch with CUDA
6. Installs all dependencies
7. Verifies CUDA is working

### Expected output at end:
```
============================================================
CUDA Verification
============================================================
PyTorch Version: 2.1.x
CUDA Available: True
CUDA Version: 12.1
GPU Device: NVIDIA L4
GPU Memory: 24.0 GB
============================================================
```

### If reboot required:
```bash
sudo reboot
# Wait 1-2 minutes, then reconnect
ssh -i dreamvision-key.pem ubuntu@<IP>
cd ~/dream_vision
./setup_aws.sh  # Run again to complete
```

---

## 6. Process Videos

### Activate Environment
```bash
cd ~/dream_vision
source venv/bin/activate
```

### Basic Usage
```bash
# Process video with default settings (768x768, L4 optimized)
python dream_vision_cloud.py --input input.mp4 --output output.mp4
```

### Advanced Options
```bash
# Higher resolution (1024x1024) - L4 can handle this
python dream_vision_cloud.py \
    --input input.mp4 \
    --output output_hd.mp4 \
    --width 1024 \
    --height 1024

# Custom prompt
python dream_vision_cloud.py \
    --input input.mp4 \
    --output output_cyber.mp4 \
    --prompt "cyberpunk neon city, glowing lights, futuristic"

# Adjust strength (0.0 = original, 1.0 = fully transformed)
python dream_vision_cloud.py \
    --input input.mp4 \
    --output output_subtle.mp4 \
    --strength 0.3

# More inference steps (better quality, slower)
python dream_vision_cloud.py \
    --input input.mp4 \
    --output output_quality.mp4 \
    --steps 4
```

### Full Command Reference
```bash
python dream_vision_cloud.py --help

Options:
  --input, -i      Input video file (default: input.mp4)
  --output, -o     Output video file (default: output_dreamvision.mp4)
  --mode, -m       Input mode: video, webcam, stream (default: video)
  --width, -W      Processing width (default: 768)
  --height, -H     Processing height (default: 768)
  --steps, -s      Inference steps, 1-4 (default: 2)
  --strength, -S   Denoising strength 0.0-1.0 (default: 0.50)
  --prompt, -p     Custom prompt for style transfer
  --seed           Random seed (default: 42)
  --fps            Output FPS (default: 30)
  --no-headless    Show display window (requires X11)
  --model          Model ID (default: stabilityai/sdxl-turbo)
```

### Monitor Progress
The script prints progress every 2 seconds:
```
[Progress] 45.2% | FPS: 28.3 | Latency: 35ms | Frames: 542 | Elapsed: 19s
```

### Monitor GPU Usage (in another terminal)
```bash
# Connect in new terminal
ssh -i dreamvision-key.pem ubuntu@<IP>

# Watch GPU usage
watch -n 1 nvidia-smi

# Or use nvtop for better visualization
nvtop
```

---

## 7. Download Results

### Method 1: SCP
```bash
# From your local machine
scp -i dreamvision-key.pem ubuntu@<IP>:~/dream_vision/output.mp4 ./

# Download multiple files
scp -i dreamvision-key.pem "ubuntu@<IP>:~/dream_vision/*.mp4" ./
```

### Method 2: AWS S3
```bash
# On EC2 instance - upload to S3
aws s3 cp ~/dream_vision/output.mp4 s3://your-bucket/results/

# On local machine - download from S3
aws s3 cp s3://your-bucket/results/output.mp4 ./
```

---

## 8. Cost Optimization

### Use Spot Instances (60-90% savings)
- Request Spot Instance when launching
- Set interruption behavior to "Stop"
- Use "persistent" request for long jobs

### Stop Instance When Not Using
```bash
# From AWS Console or CLI
aws ec2 stop-instances --instance-ids i-xxxxx
```
**Note:** Stopped instances don't charge for compute, only storage.

### Use Smaller Instance for Setup
1. Launch t3.medium for initial setup
2. Create AMI after setup complete
3. Launch g6.xlarge from AMI for processing

### Estimated Costs

| Task | Instance | Duration | Cost |
|------|----------|----------|------|
| Setup | g6.xlarge | 30 min | ~$0.40 |
| 1 min video | g6.xlarge | ~2 min | ~$0.03 |
| 10 min video | g6.xlarge | ~20 min | ~$0.27 |
| 1 hour video | g6.xlarge | ~2 hours | ~$1.60 |

**With Spot Instances:** Reduce above costs by 60-90%!

---

## 9. Troubleshooting

### CUDA Not Available
```bash
# Check NVIDIA driver
nvidia-smi

# If not found, install drivers
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Out of Memory
```bash
# Reduce resolution
python dream_vision_cloud.py --input input.mp4 --width 512 --height 512

# Or reduce batch processing
# Edit dream_vision_cloud.py: batch_size = 1
```

### Slow Performance
```bash
# Check GPU utilization
nvidia-smi

# If GPU util < 80%, check:
# - Input video resolution (larger = more CPU work)
# - Disk I/O (use SSD storage)
```

### Connection Lost During Processing
```bash
# Use screen or tmux to keep process running
screen -S dreamvision
python dream_vision_cloud.py --input input.mp4

# Detach: Ctrl+A, then D
# Reattach: screen -r dreamvision
```

### Permission Denied
```bash
# Fix script permissions
chmod +x setup_aws.sh
chmod +x run_cloud.sh
```

---

## Quick Reference Commands

```bash
# Connect to instance
ssh -i dreamvision-key.pem ubuntu@<IP>

# Upload files
scp -i dreamvision-key.pem file.mp4 ubuntu@<IP>:~/dream_vision/

# Download results
scp -i dreamvision-key.pem ubuntu@<IP>:~/dream_vision/output.mp4 ./

# Activate environment
cd ~/dream_vision && source venv/bin/activate

# Process video
python dream_vision_cloud.py --input input.mp4 --output output.mp4

# Monitor GPU
nvidia-smi -l 1

# Check disk space
df -h

# Check memory
free -h
```

---

## Expected Performance on G6 (L4 GPU)

| Resolution | Steps | Strength | FPS | Quality |
|------------|-------|----------|-----|---------|
| 512x512 | 1 | 0.45 | 35-45 | Good |
| 768x768 | 1 | 0.45 | 25-35 | Better |
| 768x768 | 2 | 0.50 | 18-25 | Best |
| 1024x1024 | 1 | 0.45 | 15-20 | HD Good |
| 1024x1024 | 2 | 0.50 | 10-15 | HD Best |

---

## Support

For issues:
1. Check GPU status: `nvidia-smi`
2. Check logs for errors
3. Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
4. Check disk space: `df -h`

---

**Enjoy creating AI art in the cloud!**

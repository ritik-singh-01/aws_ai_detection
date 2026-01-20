#!/bin/bash
################################################################################
# Dream Vision - Cloud Run Script
# Easy launcher for AWS G6 (L4 GPU)
################################################################################

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Project directory
PROJECT_DIR=~/dream_vision

echo ""
echo "============================================================"
echo -e "${GREEN}  Dream Vision - AWS Cloud Runner${NC}"
echo "============================================================"
echo ""

# Check if in project directory
if [ ! -f "$PROJECT_DIR/dream_vision_cloud.py" ]; then
    echo -e "${RED}[Error] dream_vision_cloud.py not found!${NC}"
    echo "Make sure you've uploaded the files to $PROJECT_DIR"
    exit 1
fi

cd $PROJECT_DIR

# Check virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}[Error] Virtual environment not found!${NC}"
    echo "Run setup_aws.sh first."
    exit 1
fi

# Activate virtual environment
echo "[1/3] Activating virtual environment..."
source venv/bin/activate

# Check CUDA
echo "[2/3] Checking CUDA availability..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

if [ $? -ne 0 ]; then
    echo -e "${RED}[Error] CUDA not available!${NC}"
    exit 1
fi

# Check input file
INPUT_FILE="${1:-input.mp4}"
OUTPUT_FILE="${2:-output_dreamvision.mp4}"

if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${YELLOW}[Warning] Input file not found: $INPUT_FILE${NC}"
    echo ""
    echo "Usage: ./run_cloud.sh <input_video> [output_video]"
    echo ""
    echo "Examples:"
    echo "  ./run_cloud.sh input.mp4"
    echo "  ./run_cloud.sh input.mp4 output.mp4"
    echo "  ./run_cloud.sh my_video.mp4 my_output.mp4"
    echo ""
    echo "Available video files:"
    ls -la *.mp4 2>/dev/null || echo "  No .mp4 files found"
    exit 1
fi

# Run Dream Vision
echo "[3/3] Starting Dream Vision..."
echo ""
echo "  Input:  $INPUT_FILE"
echo "  Output: $OUTPUT_FILE"
echo ""
echo "============================================================"
echo "  Processing... (Press Ctrl+C to stop)"
echo "============================================================"
echo ""

python dream_vision_cloud.py --input "$INPUT_FILE" --output "$OUTPUT_FILE"

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Output file: $OUTPUT_FILE"
echo ""
echo "To download to your local machine:"
echo "  scp -i your-key.pem ubuntu@<IP>:$PROJECT_DIR/$OUTPUT_FILE ./"
echo ""

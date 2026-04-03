#!/bin/bash
# Setup script for LivePortrait on GPU instance

set -e

# Update system
sudo apt-get update
sudo apt-get install -y ffmpeg git python3-pip python3-venv

# Clone LivePortrait
cd /home/ubuntu
git clone https://github.com/KwaiVGI/LivePortrait.git
cd LivePortrait

# Create venv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install LivePortrait requirements
pip install -r requirements_base.txt
pip install onnxruntime-gpu

# Download model weights
pip install huggingface_hub
huggingface-cli download KlingTeam/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"

# Create marker file
touch /home/ubuntu/SETUP_COMPLETE

echo "Setup complete! LivePortrait is ready."

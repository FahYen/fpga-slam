#!/usr/bin/env bash
# Launch the Vitis AI 3.0 Docker container with workspace mounted.
#
# Usage (on AWS GPU instance):
#   bash fpga/vitis/docker_run.sh
#
# Prerequisites:
#   - Docker installed with NVIDIA Container Toolkit
#   - Vitis AI image pulled: docker pull xilinx/vitis-ai-pytorch-gpu:3.0.0
#
# The script mounts:
#   /workspace  -> /workspace  (repo, models, data, output)
#
# Adjust WORKSPACE_ROOT below if your layout differs.

set -euo pipefail

IMAGE="xilinx/vitis-ai-pytorch-gpu:3.0.0"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"

# Check Docker is available
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Install Docker first."
    exit 1
fi

# Check image is pulled
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Image $IMAGE not found locally. Pulling ..."
    docker pull "$IMAGE"
fi

# Check GPU availability
GPU_FLAG=""
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_FLAG="--gpus all"
    echo "GPU detected — enabling GPU passthrough"
else
    echo "WARNING: No GPU detected — running CPU-only (calibration will be slow)"
fi

echo ""
echo "Launching Vitis AI 3.0 Docker ..."
echo "  Image:     $IMAGE"
echo "  Workspace: $WORKSPACE_ROOT"
echo "  GPU:       ${GPU_FLAG:-none}"
echo ""
echo "Inside the container, run:"
echo "  cd /workspace/slam"
echo "  python fpga/vitis/quantize_vitisai.py --help"
echo ""

# shellcheck disable=SC2086
docker run -it $GPU_FLAG \
    -v "$WORKSPACE_ROOT":/workspace \
    -w /workspace/slam \
    -e HOME=/workspace \
    "$IMAGE" \
    bash

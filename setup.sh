#!/bin/bash
# ============================================================================
# Circuit Training (AlphaChip) - Docker GPU Setup Script
# ============================================================================
# This script builds the GPU-enabled Docker image for Circuit Training.
#
# Prerequisites:
#   - Docker Desktop installed and running
#   - NVIDIA GPU drivers installed
#   - NVIDIA Container Toolkit configured (Docker Desktop handles this on Windows)
#
# Usage:
#   bash setup.sh
# ============================================================================

set -e

# --- Configuration ---
CT_VERSION="0.0.4"
PYTHON_VERSION="python3.9"
DREAMPLACE_PATTERN="dreamplace_20231214_c5a83e5_${PYTHON_VERSION}.tar.gz"
TF_AGENTS_PIP_VERSION="tf-agents[reverb]"
PLACEMENT_COST_BINARY="plc_wrapper_main_${CT_VERSION}"
IMAGE_TAG="circuit_training:gpu"

echo "============================================"
echo "  Circuit Training GPU Docker Setup"
echo "============================================"
echo ""
echo "Configuration:"
echo "  CT Version:        ${CT_VERSION}"
echo "  Python:            ${PYTHON_VERSION}"
echo "  DREAMPlace:        ${DREAMPLACE_PATTERN}"
echo "  TF-Agents:         ${TF_AGENTS_PIP_VERSION}"
echo "  PLC Binary:        ${PLACEMENT_COST_BINARY}"
echo "  Docker Image Tag:  ${IMAGE_TAG}"
echo ""

# --- Find the script's directory (where Dockerfile.gpu lives) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building Docker image from: ${SCRIPT_DIR}/Dockerfile.gpu"
echo "This will take 10-20 minutes..."
echo ""

docker build \
    --no-cache \
    --tag "${IMAGE_TAG}" \
    --build-arg base_image="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04" \
    --build-arg python_version="${PYTHON_VERSION}" \
    --build-arg tf_agents_version="${TF_AGENTS_PIP_VERSION}" \
    --build-arg dreamplace_version="${DREAMPLACE_PATTERN}" \
    --build-arg placement_cost_binary="${PLACEMENT_COST_BINARY}" \
    -f "${SCRIPT_DIR}/Dockerfile.gpu" \
    "${SCRIPT_DIR}"

echo ""
echo "============================================"
echo "  Build Complete!"
echo "============================================"
echo ""
echo "To verify GPU access:"
echo "  docker run --gpus all --rm ${IMAGE_TAG} nvidia-smi"
echo ""
echo "To verify TensorFlow GPU detection:"
echo "  docker run --gpus all --rm ${IMAGE_TAG} python3.9 -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
echo ""
echo "To run the toy netlist example:"
echo "  bash run_toy_example.sh"
echo ""

#!/bin/bash
# ============================================================================
# Circuit Training (AlphaChip) - Run Toy Netlist Example (GPU)
# ============================================================================
# Runs the end-to-end smoke test on the Ariane RISC-V toy netlist using GPU.
# This starts a Reverb server, collect jobs, and trains for 1 iteration.
#
# Prerequisites:
#   - Docker image "circuit_training:gpu" built (run setup.sh first)
#   - NVIDIA GPU available
#
# Usage:
#   bash run_toy_example.sh
#
# Expected runtime: 10-30 minutes depending on GPU
# ============================================================================

set -e

IMAGE_TAG="circuit_training:gpu"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/circuit_training"

# Netlist files (Ariane RISC-V CPU - toy example included in the repo)
NETLIST_FILE="./circuit_training/environment/test_data/ariane/netlist.pb.txt"
INIT_PLACEMENT="./circuit_training/environment/test_data/ariane/initial.plc"

# Output directory on the host
OUTPUT_DIR="${REPO_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "  Circuit Training - Toy Netlist (GPU)"
echo "============================================"
echo ""
echo "Image:          ${IMAGE_TAG}"
echo "Netlist:        ${NETLIST_FILE}"
echo "Init Placement: ${INIT_PLACEMENT}"
echo "Output Dir:     ${OUTPUT_DIR}"
echo ""

# --- Step 1: Quick GPU check ---
echo "--- Verifying GPU access ---"
docker run --gpus all --rm "${IMAGE_TAG}" nvidia-smi
echo ""

# --- Step 2: Quick TensorFlow GPU check ---
echo "--- Verifying TensorFlow GPU detection ---"
docker run --gpus all --rm "${IMAGE_TAG}" \
    python3.9 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'TensorFlow sees {len(gpus)} GPU(s): {gpus}')"
echo ""

# --- Step 3: Run the end-to-end smoke test ---
echo "--- Starting end-to-end smoke test with GPU ---"
echo "This will take 10-30 minutes..."
echo "Logs will be written to: ${OUTPUT_DIR}"
echo ""

docker run --gpus all --rm \
    -v "${REPO_DIR}":/workspace \
    --workdir /workspace \
    "${IMAGE_TAG}" \
    bash tools/e2e_smoke_test.sh \
        --root_dir /workspace/logs \
        --use_gpu True \
        --netlist_file "${NETLIST_FILE}" \
        --init_place "${INIT_PLACEMENT}"

echo ""
echo "============================================"
echo "  Smoke Test Complete!"
echo "============================================"
echo ""
echo "Output files are in: ${OUTPUT_DIR}"
echo ""
echo "Expected output structure:"
echo "  logs/"
echo "    ├── collect_*.log     # Collect job logs"
echo "    ├── reverb.log        # Reverb server log"
echo "    ├── run_00/           # Training run directory"
echo "    │   ├── train/        # TF event files (TensorBoard)"
echo "    │   ├── policies/     # Saved policy checkpoints"
echo "    │   └── replay_buffer # Reverb replay data"
echo ""
echo "To view training metrics in TensorBoard:"
echo "  pip install tensorboard"
echo "  tensorboard --logdir ${OUTPUT_DIR}"
echo ""

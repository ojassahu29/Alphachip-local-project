#!/bin/bash
# Runs the Circuit Training end-to-end smoke test using our Python plc_wrapper stub.
# This is needed because the official plc_wrapper_main binary from GCS
# is no longer publicly accessible (HTTP 403 Forbidden).

set -e

ROOT_DIR=${1:-/workspace/logs/run_00}
NETLIST_FILE=/workspace/circuit_training/environment/test_data/ariane/netlist.pb.txt
INIT_PLACEMENT=/workspace/circuit_training/environment/test_data/ariane/initial.plc
REVERB_PORT=8008
REVERB_SERVER=127.0.0.1:${REVERB_PORT}
NUM_COLLECT_JOBS=2
SCRIPT_LOGS=${ROOT_DIR}
PLC_STUB=/workspace/plc_wrapper_stub.py

mkdir -p ${ROOT_DIR}

echo "================================================"
echo "  Circuit Training Toy Netlist (Ariane RISC-V)"
echo "================================================"
echo "Root dir:  ${ROOT_DIR}"
echo "Netlist:   ${NETLIST_FILE}"
echo "PLC stub:  ${PLC_STUB}"
echo ""

# Verify stub is available
python3.9 ${PLC_STUB} --pipe_address=/dev/null --netlist_file=${NETLIST_FILE} &
PLC_TEST_PID=$!
sleep 1
kill ${PLC_TEST_PID} 2>/dev/null || true
echo "PLC stub: OK"

# Start Reverb server in background
echo "Starting Reverb server..."
CUDA_VISIBLE_DEVICES=-1 python3.9 -m circuit_training.learning.ppo_reverb_server \
  --root_dir=${ROOT_DIR} --port=${REVERB_PORT} \
  &> ${SCRIPT_LOGS}/reverb.log &
REVERB_PID=$!
echo "Reverb server PID: ${REVERB_PID}"

# Give reverb time to start
sleep 10

# Start collect jobs
echo "Starting ${NUM_COLLECT_JOBS} collect jobs..."
for i in $(seq 1 ${NUM_COLLECT_JOBS}); do
  CUDA_VISIBLE_DEVICES=-1 python3.9 -m circuit_training.learning.ppo_collect \
    --root_dir=${ROOT_DIR} \
    --std_cell_placer_mode=dreamplace \
    --replay_buffer_server_address=${REVERB_SERVER} \
    --variable_container_server_address=${REVERB_SERVER} \
    --task_id=0 \
    --netlist_file=${NETLIST_FILE} \
    --init_placement=${INIT_PLACEMENT} \
    --plc_wrapper_main="python3.9 ${PLC_STUB}" \
    &> ${SCRIPT_LOGS}/collect_${i}.log &
  echo "Collect job ${i} PID: $!"
done

echo "Starting training job (1 iteration)..."
python3.9 -m circuit_training.learning.train_ppo \
  --root_dir=${ROOT_DIR} \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --std_cell_placer_mode=dreamplace \
  --gin_bindings='train.per_replica_batch_size=5' \
  --gin_bindings='train.num_iterations=1' \
  --gin_bindings='train.num_episodes_per_iteration=5' \
  --gin_bindings='train.num_epochs=4' \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT} \
  --plc_wrapper_main="python3.9 ${PLC_STUB}" \
  --use_gpu=False

echo "================================================"
echo "  Smoke Test COMPLETE!"
echo "  Logs in: ${ROOT_DIR}"
echo "================================================"

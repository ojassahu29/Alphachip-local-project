# Circuit Training (AlphaChip) — GPU Docker Setup Guide

Complete guide to run Google Research's [Circuit Training](https://github.com/google-research/circuit_training) on your Windows machine using Docker with **NVIDIA GPU acceleration**.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Windows 10/11 with WSL2 |
| **Docker Desktop** | Latest version with WSL2 backend enabled |
| **NVIDIA GPU** | Any CUDA-capable GPU |
| **NVIDIA Drivers** | Latest Game Ready or Studio drivers |
| **Disk Space** | ~15 GB for Docker image |

### Verify Docker GPU Support

Before starting, confirm GPU passthrough works:

```powershell
docker run --gpus all --rm nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 nvidia-smi
```

If this fails, ensure:
1. Docker Desktop → Settings → General → **"Use the WSL 2 based engine"** is checked
2. NVIDIA drivers are up to date
3. WSL2 is properly installed (`wsl --update`)

---

## Quick Start (3 commands)

```powershell
# From z:\Alphachip directory:

# 1. Build the Docker image (~15-20 min)
setup.bat

# 2. Run the toy netlist with GPU (~10-30 min)
run_toy_example.bat
```

That's it! The script will verify GPU access, run TensorFlow GPU detection, and execute the smoke test.

---

## Manual Commands (Step by Step)

### 1. Build the Docker Image

```powershell
cd z:\Alphachip

docker build --no-cache --tag circuit_training:gpu ^
    --build-arg base_image="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04" ^
    --build-arg tf_agents_version="tf-agents[reverb]" ^
    --build-arg dreamplace_version="dreamplace_20231214_c5a83e5_python3.9.tar.gz" ^
    --build-arg placement_cost_binary="plc_wrapper_main_0.0.4" ^
    -f Dockerfile.gpu .
```

### 2. Verify GPU Access

```powershell
docker run --gpus all --rm circuit_training:gpu nvidia-smi
```

Expected output: Your NVIDIA GPU name, driver version, CUDA 11.8.

### 3. Verify TensorFlow Sees the GPU

```powershell
docker run --gpus all --rm circuit_training:gpu python3.9 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

### 4. Run Interactive Container

```powershell
docker run --gpus all -it --rm ^
    -v z:\Alphachip\circuit_training:/workspace ^
    --workdir /workspace ^
    circuit_training:gpu bash
```

### 5. Run the Toy Netlist (Ariane RISC-V)

```powershell
docker run --gpus all --rm ^
    -v z:\Alphachip\circuit_training:/workspace ^
    --workdir /workspace ^
    circuit_training:gpu ^
    bash tools/e2e_smoke_test.sh --root_dir /workspace/logs --use_gpu True
```

---

## Toy Netlist Details

The included toy netlist is the **Ariane RISC-V CPU**, located at:

| File | Path |
|---|---|
| Netlist | `circuit_training/environment/test_data/ariane/netlist.pb.txt` |
| Initial Placement | `circuit_training/environment/test_data/ariane/initial.plc` |

The smoke test (`e2e_smoke_test.sh`) does the following:
1. Starts a **Reverb replay buffer server** on port 8008
2. Launches **4 collect jobs** that generate placement episodes using DREAMPlace
3. Runs **PPO training** for 1 iteration (5 episodes, 4 epochs)
4. Uses GPU for the training step (`--use_gpu True`)

---

## Expected Output Files

After the smoke test completes, check `z:\Alphachip\circuit_training\logs\`:

```
logs/
├── reverb.log              # Reverb server log
├── collect_1.log           # Collect job 1 log
├── collect_2.log           # Collect job 2 log
├── collect_3.log           # Collect job 3 log
├── collect_4.log           # Collect job 4 log
└── run_00/                 # Training output directory
    ├── train/              # TensorBoard event files
    │   └── events.out.tfevents.*
    ├── policies/           # Saved policy checkpoints
    │   ├── greedy_policy/
    │   └── collect_policy/
    └── replay_buffer/      # Reverb replay data
```

### Key Output Descriptions

| Output | Description |
|---|---|
| **Event files** (`train/events.out.tfevents.*`) | TensorBoard logs with training metrics: reward, loss, wirelength, congestion, density |
| **Greedy policy** (`policies/greedy_policy/`) | The trained placement policy (SavedModel format) |
| **Collect policy** (`policies/collect_policy/`) | The exploration policy used during data collection |
| **Collect logs** (`collect_*.log`) | Per-episode placement metrics and DREAMPlace output |
| **Reverb log** (`reverb.log`) | Replay buffer server status |

### View Training Metrics

```powershell
pip install tensorboard
tensorboard --logdir z:\Alphachip\circuit_training\logs
```

---

## Dependency Versions

| Package | Version | Purpose |
|---|---|---|
| CUDA | 11.8 | GPU compute |
| cuDNN | 8.6 (devel) | Deep learning primitives |
| Python | 3.9 | Runtime |
| TensorFlow | 2.x (stable) | ML framework |
| TF-Agents | stable | RL agent framework |
| Reverb | stable (via tf-agents) | Replay buffer |
| DREAMPlace | 20231214_c5a83e5 | Standard cell placement |
| PyTorch | 1.13.1 | Required by DREAMPlace |
| Keras | 2.x (legacy) | TF backend |
| Placement Cost | plc_wrapper_main_0.0.4 | Cost evaluation binary |

---

## File Structure

```
z:\Alphachip\
├── Dockerfile.gpu          # GPU-enabled Dockerfile
├── setup.bat               # Windows build script
├── setup.sh                # Linux/WSL build script
├── run_toy_example.bat     # Windows: run toy netlist
├── run_toy_example.sh      # Linux/WSL: run toy netlist
├── README_GPU_SETUP.md     # This file
└── circuit_training/       # Cloned repo (r0.0.4 branch)
    ├── circuit_training/
    │   ├── environment/
    │   │   └── test_data/
    │   │       └── ariane/         # Toy netlist
    │   │           ├── netlist.pb.txt
    │   │           └── initial.plc
    │   ├── learning/
    │   │   ├── train_ppo.py        # Training script
    │   │   ├── ppo_collect.py      # Data collection
    │   │   └── ppo_reverb_server.py
    │   └── grouping/
    ├── tools/
    │   ├── e2e_smoke_test.sh       # End-to-end test
    │   └── docker/                 # Original Dockerfiles
    └── docs/
        └── ARIANE.md               # Full Ariane example
```

---

## Troubleshooting

### "docker: Error response from daemon: could not select device driver"
→ NVIDIA Container Toolkit is not set up. On Windows with Docker Desktop + WSL2, update your NVIDIA drivers and restart Docker Desktop.

### TensorFlow doesn't see GPU
→ Make sure you use `--gpus all` flag. Check `nvidia-smi` works first. The base image must be `devel` (not `runtime`) for XLA support.

### DREAMPlace import errors
→ The `PYTHONPATH` is set in the Dockerfile. If running locally, ensure `/dreamplace` and `/dreamplace/dreamplace` are on the path.

### Build fails downloading files
→ The Google Cloud Storage URLs for DREAMPlace and plc_wrapper_main occasionally have connectivity issues. Retry the build.

### Out of GPU memory
→ The smoke test uses modest GPU memory. If you have a GPU with < 4GB VRAM, try reducing batch size:
```
--gin_bindings='train.per_replica_batch_size=2'
```

@echo off
REM ============================================================================
REM Circuit Training (AlphaChip) - Run Toy Netlist Example (Windows/GPU)
REM ============================================================================
REM Runs the end-to-end smoke test on the Ariane RISC-V toy netlist using GPU.
REM
REM Prerequisites:
REM   - Docker image "circuit_training:gpu" built (run setup.bat first)
REM   - NVIDIA GPU available
REM
REM Usage:
REM   run_toy_example.bat
REM
REM Expected runtime: 10-30 minutes depending on GPU
REM ============================================================================

set IMAGE_TAG=circuit_training:gpu
set REPO_DIR=%~dp0circuit_training

echo ============================================
echo   Circuit Training - Toy Netlist (GPU)
echo ============================================
echo.

REM Create output directory
if not exist "%REPO_DIR%\logs" mkdir "%REPO_DIR%\logs"

echo --- Step 1: Verifying GPU access ---
docker run --gpus all --rm %IMAGE_TAG% nvidia-smi
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: GPU not accessible! Make sure:
    echo   1. NVIDIA drivers are installed
    echo   2. Docker Desktop WSL2 backend is enabled
    echo   3. Docker Desktop has "Use the WSL 2 based engine" checked
    pause
    exit /b 1
)
echo.

echo --- Step 2: Verifying TensorFlow GPU detection ---
docker run --gpus all --rm %IMAGE_TAG% python3.9 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'TensorFlow sees {len(gpus)} GPU(s): {gpus}')"
echo.

echo --- Step 3: Running end-to-end smoke test with GPU ---
echo This will take 10-30 minutes...
echo Logs will be written to: %REPO_DIR%\logs
echo.

docker run --gpus all --rm ^
    -v "%REPO_DIR%":/workspace ^
    --workdir /workspace ^
    %IMAGE_TAG% ^
    bash tools/e2e_smoke_test.sh ^
        --root_dir /workspace/logs ^
        --use_gpu True

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Smoke test failed! Check logs in %REPO_DIR%\logs
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Smoke Test Complete!
echo ============================================
echo.
echo Output files are in: %REPO_DIR%\logs
echo.
echo Expected output structure:
echo   logs\
echo     +-- collect_*.log      (Collect job logs)
echo     +-- reverb.log         (Reverb server log)
echo     +-- run_00\            (Training run directory)
echo         +-- train\         (TF event files for TensorBoard)
echo         +-- policies\      (Saved policy checkpoints)
echo.
pause

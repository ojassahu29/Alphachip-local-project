@echo off
REM ============================================================================
REM Circuit Training (AlphaChip) - Docker GPU Setup Script (Windows)
REM ============================================================================
REM This script builds the GPU-enabled Docker image for Circuit Training.
REM
REM Prerequisites:
REM   - Docker Desktop installed and running
REM   - NVIDIA GPU drivers installed
REM   - WSL2 backend enabled in Docker Desktop
REM
REM Usage:
REM   setup.bat
REM ============================================================================

set CT_VERSION=0.0.4
set PYTHON_VERSION=python3.9
set DREAMPLACE_PATTERN=dreamplace_20231214_c5a83e5_%PYTHON_VERSION%.tar.gz
set TF_AGENTS_PIP_VERSION=tf-agents[reverb]
set PLACEMENT_COST_BINARY=plc_wrapper_main_%CT_VERSION%
set IMAGE_TAG=circuit_training:gpu

echo ============================================
echo   Circuit Training GPU Docker Setup
echo ============================================
echo.
echo Configuration:
echo   CT Version:        %CT_VERSION%
echo   Python:            %PYTHON_VERSION%
echo   DREAMPlace:        %DREAMPLACE_PATTERN%
echo   TF-Agents:         %TF_AGENTS_PIP_VERSION%
echo   PLC Binary:        %PLACEMENT_COST_BINARY%
echo   Docker Image Tag:  %IMAGE_TAG%
echo.
echo Building Docker image... This will take 10-20 minutes.
echo.

docker build ^
    --no-cache ^
    --tag %IMAGE_TAG% ^
    --build-arg base_image="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04" ^
    --build-arg python_version="%PYTHON_VERSION%" ^
    --build-arg tf_agents_version="%TF_AGENTS_PIP_VERSION%" ^
    --build-arg dreamplace_version="%DREAMPLACE_PATTERN%" ^
    --build-arg placement_cost_binary="%PLACEMENT_COST_BINARY%" ^
    -f "%~dp0Dockerfile.gpu" ^
    "%~dp0."

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Docker build failed!
    echo Make sure Docker Desktop is running and has WSL2 backend enabled.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Build Complete!
echo ============================================
echo.
echo To verify GPU access:
echo   docker run --gpus all --rm %IMAGE_TAG% nvidia-smi
echo.
echo To run the toy netlist example:
echo   run_toy_example.bat
echo.
pause

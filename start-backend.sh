#!/bin/bash
# VocalBlade Backend Startup Script
# Includes CUDA 11 libraries needed by onnxruntime/pyannote alongside CUDA 12 for PyTorch

ROOT="$(cd "$(dirname "$0")" && pwd)"

export LD_LIBRARY_PATH="$ROOT/vocal_blade_backend/.cudnn8/lib:$LD_LIBRARY_PATH"
source "$HOME/.local/bin/env"
cd "$ROOT/vocal_blade_backend/src"
exec uv run python manage.py runserver 0.0.0.0:8000 --noreload

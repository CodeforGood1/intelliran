#!/usr/bin/env bash
# setup.sh - Create venv and install dependencies (WSL-aware)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== RAN AI Lab Setup ==="

# Detect OS
if grep -qi microsoft /proc/version 2>/dev/null || [ "$(uname -s)" = "Linux" ]; then
    echo "Running on Linux/WSL"
    PYTHON=python3
else
    echo "Running on Windows (use WSL for Mininet)"
    PYTHON=python
fi

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate
source venv/bin/activate
echo "Python: $(python --version)"

# Upgrade pip
pip install --upgrade pip --quiet

# Check CUDA
CUDA_AVAIL=$(python -c "
try:
    import torch
    print('yes' if torch.cuda.is_available() else 'no')
except:
    print('no')
" 2>/dev/null || echo "no")

echo "CUDA available: $CUDA_AVAIL"

if [ "$CUDA_AVAIL" = "yes" ]; then
    echo "Installing GPU torch wheel..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
else
    echo "Installing CPU-only torch (saves ~1.5GB)..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
fi

# Install remaining deps
echo "Installing dependencies..."
pip install numpy pandas scikit-learn lightgbm onnx onnxruntime lime shap \
    fastapi uvicorn[standard] streamlit psutil pyyaml pytest requests pyarrow joblib --quiet

echo "=== Setup complete ==="
echo "Activate with: source venv/bin/activate"

#!/bin/bash
# install.sh — One-command setup for LLM EdgeCraft on Linux/Mac
# Usage: ./scripts/install.sh
# Requires: Python 3.11+, Node.js 18+, NVIDIA GPU + CUDA 12+ (for training)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$ROOT/backend/.venv"

echo ""
echo "========================================"
echo "  LLM EdgeCraft — Installation"
echo "========================================"
echo ""

# 1. Check Python
echo "[1/6] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 not found. Install Python 3.11+" && exit 1
fi
python3 --version

# 2. Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    echo "      Created $VENV"
else
    echo "      Already exists: $VENV"
fi
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

# 3. Install PyTorch with CUDA detection
echo "[3/6] Installing PyTorch..."
if command -v nvidia-smi &>/dev/null; then
    echo "      NVIDIA GPU detected — installing CUDA version"
    $PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
else
    echo "      No GPU — installing CPU-only PyTorch"
    $PIP install torch torchvision torchaudio -q
fi

# 4. Install Unsloth
echo "[4/6] Installing Unsloth (+ transformers, trl, peft, datasets)..."
$PIP install unsloth -q

# 5. Install backend deps
echo "[5/6] Installing backend dependencies..."
$PIP install -r "$ROOT/backend/requirements.txt" -q

# 6. Install frontend deps
echo "[6/6] Installing frontend dependencies..."
cd "$ROOT/frontend" && npm install --silent
cd "$ROOT"

# Setup .env
if [ ! -f "$ROOT/backend/.env" ]; then
    cp "$ROOT/backend/.env.example" "$ROOT/backend/.env"
    echo ""
    echo "Created backend/.env — please edit and set a strong JWT_SECRET"
fi

echo ""
echo "========================================"
echo "  Installation complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Edit backend/.env and set JWT_SECRET"
echo "  2. Demo mode: ./scripts/start-dev.sh"
echo "  3. Real mode: ./scripts/start-dev.sh --real"
echo ""
echo "Use the venv Python for the backend:"
echo "  $PYTHON backend/run.py"
echo ""

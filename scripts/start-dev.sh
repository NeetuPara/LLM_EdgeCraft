#!/bin/bash
# start-dev.sh — Start both backend and frontend for development
# Usage: ./scripts/start-dev.sh [--real | --demo]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

# Find the right Python: prefer backend/.venv, fall back to system python
if [ -f "$BACKEND/.venv/bin/python" ]; then
    PYTHON="$BACKEND/.venv/bin/python"
    echo "Using venv: backend/.venv"
elif [ -f "$BACKEND/.venv/Scripts/python.exe" ]; then
    # Git Bash on Windows
    PYTHON="$BACKEND/.venv/Scripts/python.exe"
    echo "Using venv: backend/.venv (Windows)"
else
    PYTHON="python3"
    echo "Warning: backend/.venv not found. Run ./scripts/install.sh first."
fi

MODE="demo"
[[ "$1" == "--real" ]] && MODE="real"

echo ""
echo "========================================"
echo "  LLM EdgeCraft — $MODE mode"
echo "========================================"
echo ""

if [ "$MODE" = "real" ]; then
    echo "[1/2] Starting backend (port 8001)..."
    cd "$BACKEND"
    "$PYTHON" run.py &
    BACKEND_PID=$!
    echo "      Backend PID: $BACKEND_PID"
    sleep 4
fi

echo "[2/2] Starting frontend (port 5174)..."
cd "$FRONTEND"
npm run "dev:$MODE" &
FRONTEND_PID=$!

echo ""
echo "  Frontend: http://localhost:5174"
[ "$MODE" = "real" ] && echo "  Backend:  http://localhost:8001"
[ "$MODE" = "real" ] && echo "  API docs: http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

wait

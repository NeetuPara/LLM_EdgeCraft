# start-dev.ps1 — Start both backend and frontend for development
# Usage: .\scripts\start-dev.ps1 [--real | --demo]

param(
    [switch]$real,
    [switch]$demo
)

$ROOT     = Split-Path -Parent $PSScriptRoot
$BACKEND  = Join-Path $ROOT "backend"
$FRONTEND = Join-Path $ROOT "frontend"

# ── Find the right Python ──
# Priority: 1) backend/.venv  2) system python (must have unsloth)
$VENV_PYTHON = Join-Path $BACKEND ".venv\Scripts\python.exe"
if (Test-Path $VENV_PYTHON) {
    $PYTHON = $VENV_PYTHON
    Write-Host "Using venv: backend/.venv" -ForegroundColor DarkGray
} else {
    $PYTHON = "python"
    Write-Host "Warning: backend/.venv not found. Run .\scripts\install.ps1 first." -ForegroundColor Yellow
    Write-Host "Falling back to system python (may not have all deps)" -ForegroundColor Yellow
}

$mode = if ($real) { "real" } else { "demo" }
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LLM EdgeCraft — $mode mode"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start backend
if ($real) {
    Write-Host "[1/2] Starting backend (port 8001)..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$BACKEND'; & '$PYTHON' run.py" -WindowStyle Normal
    Write-Host "      Waiting for backend..."
    Start-Sleep -Seconds 4
}

# Start frontend
$npmMode = if ($real) { "dev:real" } else { "dev:demo" }
Write-Host "[2/2] Starting frontend ($npmMode, port 5174)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$FRONTEND'; npm run $npmMode"

Write-Host ""
Write-Host "  Frontend: http://localhost:5174" -ForegroundColor Yellow
if ($real) {
    Write-Host "  Backend:  http://localhost:8001" -ForegroundColor Yellow
    Write-Host "  API docs: http://localhost:8001/docs" -ForegroundColor Yellow
}
Write-Host ""

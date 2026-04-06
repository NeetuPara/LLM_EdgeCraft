# install.ps1 -- EdgeCraft setup for Windows
# Usage: .\scripts\install.ps1  (run from project ROOT, not from backend\)
#
# IMPORTANT: Unsloth pulls CPU-only torch from PyPI during install.
# This script reinstalls CUDA torch as the final step to fix that.

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  EdgeCraft -- Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# --- 1. Check Python ---
Write-Host "[1/7] Checking Python..." -ForegroundColor Green
try {
    $pythonVersion = python --version 2>&1
    Write-Host "      Found: $pythonVersion"
} catch {
    Write-Host "ERROR: Python not found. Install Python 3.11+ from python.org" -ForegroundColor Red
    exit 1
}

# --- 2. Detect GPU and CUDA version (always run, needed later for torch reinstall) ---
Write-Host "[2/7] Detecting GPU..." -ForegroundColor Green
$gpuFound  = $false
$cudaTag   = "cpu"
$cudaLabel = "CPU-only"
try {
    $smiOut = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        $gpuFound = $true
        $cudaLine = ($smiOut | Select-String "CUDA Version")
        if ("$cudaLine" -match "CUDA Version:\s*(\d+)\.(\d+)") {
            $cudaNum = [int]$Matches[1] * 10 + [int]$Matches[2]
            if     ($cudaNum -ge 128) { $cudaTag = "cu128"; $cudaLabel = "CUDA 12.8 (Blackwell/RTX 50xx)" }
            elseif ($cudaNum -ge 124) { $cudaTag = "cu124"; $cudaLabel = "CUDA 12.4" }
            elseif ($cudaNum -ge 121) { $cudaTag = "cu121"; $cudaLabel = "CUDA 12.1" }
            elseif ($cudaNum -ge 118) { $cudaTag = "cu118"; $cudaLabel = "CUDA 11.8" }
            else                      { $cudaTag = "cu121"; $cudaLabel = "CUDA 12.1 (fallback)" }
        }
        Write-Host "      NVIDIA GPU detected -- $cudaLabel"
    } else {
        Write-Host "      No NVIDIA GPU found -- will use CPU PyTorch" -ForegroundColor Yellow
    }
} catch { $gpuFound = $false }

# --- 3. Create virtual environment ---
$VENV   = Join-Path $ROOT "backend\.venv"
$PYTHON = Join-Path $VENV "Scripts\python.exe"
$PIP    = Join-Path $VENV "Scripts\pip.exe"

if (-not (Test-Path $PYTHON)) {
    Write-Host "[3/7] Creating virtual environment..." -ForegroundColor Green
    python -m venv $VENV
} else {
    Write-Host "[3/7] Virtual environment already exists" -ForegroundColor Yellow
}

# Always upgrade pip (old pip has bugs with PyTorch index)
& $PYTHON -m pip install --upgrade pip --quiet

# --- 4. Install Unsloth (pulls CPU-only torch from PyPI -- expected, fixed in step 5) ---
Write-Host "[4/7] Installing Unsloth..." -ForegroundColor Green
$unslothOk = & $PYTHON -c "import unsloth; print('ok')" 2>$null
if ($unslothOk -eq "ok") {
    Write-Host "      Unsloth already installed -- skipping" -ForegroundColor Yellow
} else {
    Write-Host "      Downloading Unsloth and dependencies (may take a few minutes)..." -ForegroundColor Green
    & $PIP install unsloth --quiet
    Write-Host "      Unsloth installed"
}

# --- 5. Reinstall CUDA PyTorch (CRITICAL -- unsloth replaces it with CPU version) ---
# Unsloth resolves torch from PyPI which only has CPU builds.
# --force-reinstall --no-deps replaces it with the CUDA build without triggering
# unsloth's dependency resolver again.
Write-Host "[5/7] Installing PyTorch with CUDA ($cudaLabel)..." -ForegroundColor Green
if ($gpuFound) {
    Write-Host "      Forcing CUDA torch install (overrides CPU torch from unsloth)..." -ForegroundColor Green
    # torch 2.10.0 + torchvision 0.25.0 — correct pair, compatible with unsloth + triton 3.6
    # (torch 2.7.0 causes triton_key ImportError with triton-windows 3.6; 2.10.0 does not)
    & $PIP install "torch==2.10.0+$cudaTag" "torchvision==0.25.0+$cudaTag" "torchaudio==2.10.0+$cudaTag" `
        --index-url "https://download.pytorch.org/whl/$cudaTag" `
        --force-reinstall --no-deps --quiet
    # Verify CUDA is active
    $cudaCheck = & $PYTHON -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cudaCheck -eq "True") {
        $torchVer = & $PYTHON -c "import torch; print(torch.__version__)" 2>$null
        Write-Host "      PyTorch $torchVer with CUDA confirmed" -ForegroundColor Green
    } else {
        Write-Host "      WARNING: CUDA not working -- check CUDA toolkit installation" -ForegroundColor Red
    }
} else {
    Write-Host "      No GPU -- keeping CPU PyTorch" -ForegroundColor Yellow
}

# --- 6. Install remaining backend requirements ---
Write-Host "[6/7] Installing backend requirements..." -ForegroundColor Green
$REQ = Join-Path $ROOT "backend\requirements.txt"
& $PIP install -r $REQ --quiet
Write-Host "      Backend requirements installed"

# --- 7. Install frontend dependencies ---
Write-Host "[7/7] Installing frontend dependencies..." -ForegroundColor Green
Push-Location (Join-Path $ROOT "frontend")
npm install --silent
Pop-Location
Write-Host "      Frontend dependencies installed"

# --- Copy .env if missing ---
$ENV_FILE    = Join-Path $ROOT "backend\.env"
$ENV_EXAMPLE = Join-Path $ROOT "backend\.env.example"
if ((-not (Test-Path $ENV_FILE)) -and (Test-Path $ENV_EXAMPLE)) {
    Copy-Item $ENV_EXAMPLE $ENV_FILE
    Write-Host ""
    Write-Host "Created backend\.env -- edit it and set JWT_SECRET" -ForegroundColor Yellow
}

# --- Final verification ---
Write-Host ""
$finalCheck = & $PYTHON -c "import torch, unsloth; print(f'torch={torch.__version__} | cuda={torch.cuda.is_available()}')" 2>$null
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host "  $finalCheck" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit backend\.env and set JWT_SECRET"
Write-Host "  2. Demo mode (no backend):  .\scripts\start-dev.ps1"
Write-Host "  3. Real mode (GPU training): .\scripts\start-dev.ps1 --real"
Write-Host ""
Write-Host "Run backend manually:" -ForegroundColor Cyan
Write-Host "  cd backend"
Write-Host "  .\.venv\Scripts\python.exe run.py"
Write-Host ""

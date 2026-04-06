# EdgeCraft — LLM Fine-tuning Studio

Full-stack local LLM fine-tuning. Train any open-source model on your GPU, run inference, export to GGUF.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 or 3.12 | Must be on PATH — check with `python --version` |
| Node.js | 18+ | For frontend — check with `node --version` |
| NVIDIA GPU + CUDA | CUDA 12+ | Only needed for real training — not for demo mode |

---

## Installation (Windows — Step by Step)

> **Important:** All commands below must be run from the **project root** (`D:\EdgeCraft\LLM_EdgeCraft\`), NOT from inside `backend\` or `frontend\`.

### Step 1 — Open PowerShell as normal user (not admin)

```powershell
cd D:\EdgeCraft\LLM_EdgeCraft
```

### Step 2 — Allow PowerShell scripts to run (one-time)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3 — Run the automated installer

```powershell
.\scripts\install.ps1
```

This script will:
- Create `backend\.venv\` (Python virtual environment)
- Install PyTorch with CUDA support
- Install Unsloth + all ML dependencies
- Install FastAPI, SQLAlchemy, and all backend packages
- Run `npm install` for the frontend

> Installation takes 5-15 minutes depending on internet speed (PyTorch ~2.5 GB download).

---

## Running

### Option 1 — One command (recommended)

```powershell
# Demo mode — no GPU needed, uses mock data
.\scripts\start-dev.ps1

# Real mode — connects to real backend, enables GPU training
.\scripts\start-dev.ps1 --real
```

### Option 2 — Two terminals manually

**Terminal 1 — Backend** (run from project root or backend folder):
```powershell
cd D:\EdgeCraft\LLM_EdgeCraft\backend

# Activate the virtual environment first
.\.venv\Scripts\Activate.ps1

# Then start the server
python run.py
```

**Terminal 2 — Frontend** (run from project root or frontend folder):
```powershell
cd D:\EdgeCraft\LLM_EdgeCraft\frontend

npm run dev:demo    # demo mode (no backend needed)
npm run dev:real    # real mode (backend must be running)
```

> Always start the **backend before** the frontend when using real mode.

---

## Access

| URL | What |
|---|---|
| http://localhost:5174 | Frontend (EdgeCraft UI) |
| http://localhost:8001 | Backend API |
| http://localhost:8001/docs | Auto-generated API docs (Swagger) |

---

## Common Errors & Fixes

### `.venv\Scripts\activate` not recognized
You are either:
- In the wrong folder — make sure you are inside `backend\`, not the project root
- The `.venv` has not been created yet — run `.\scripts\install.ps1` from the project root first

```powershell
# Correct way to activate from inside backend\ folder:
.\.venv\Scripts\Activate.ps1      # PowerShell
.\.venv\Scripts\activate.bat      # CMD

# Or run Python directly without activating:
.\.venv\Scripts\python.exe run.py
```

### `ModuleNotFoundError: No module named 'dotenv'` or any other module
You are using the **system Python** instead of the venv Python. Always activate the venv first, or use `.\.venv\Scripts\python.exe run.py` directly.

### `install.ps1 not recognized`
You are running from the wrong folder. The install script is at `scripts\install.ps1` inside the **project root**, not inside `backend\`.

```powershell
# Wrong (from inside backend\):
.\install.ps1

# Correct (from project root D:\EdgeCraft\LLM_EdgeCraft\):
.\scripts\install.ps1
```

### `CUDA Out of Memory` during training
- Reduce `batch_size` to 1
- Reduce `max_seq_length` to 1024
- Use QLoRA instead of LoRA (4-bit quantization)
- Reduce `lora_r` from 64 to 32

### Frontend shows blank screen
Check browser console (F12). Usually a Vite proxy error meaning the backend is not running. Start backend first.

---

## Manual Setup (if install.ps1 fails)

> **Critical:** Unsloth pulls CPU-only torch from PyPI during install. You **must** reinstall CUDA torch **after** unsloth or training will fail with "no GPU found".

```powershell
# ── Step 1: Go to project root ──
cd D:\EdgeCraft\LLM_EdgeCraft

# ── Step 2: Create virtual environment ──
python -m venv backend\.venv

# ── Step 3: Activate venv ──
backend\.venv\Scripts\Activate.ps1

# ── Step 4: Upgrade pip (old pip has a bug with the PyTorch index) ──
python -m pip install --upgrade pip

# ── Step 5: Install Unsloth ──
# Installs: transformers, trl, peft, datasets, xformers, bitsandbytes, triton
# WARNING: also installs CPU-only torch from PyPI — fixed in next step
pip install unsloth

# ── Step 6: Reinstall CUDA PyTorch AFTER unsloth (CRITICAL) ──
# Unsloth replaces the CUDA torch with CPU-only during its install.
# --force-reinstall puts CUDA torch back.
# --no-deps skips re-resolution so unsloth can't pull CPU torch again.
#
# RTX 50xx (Blackwell, CUDA 12.8):
pip install "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.10.0+cu128" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
# torch 2.10.0 + torchvision 0.25.0 -- correct pair, compatible with unsloth + triton 3.6
# (torch 2.7.0 causes triton_key ImportError with triton-windows 3.6; 2.10.0 does not)
#
# RTX 40xx / 30xx (CUDA 12.4):
# pip install "torch==2.7.0+cu124" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
#
# RTX 30xx / 20xx (CUDA 12.1):
# pip install "torch==2.7.0+cu121" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps

# ── Step 7: Install remaining backend requirements ──
pip install -r backend\requirements.txt

# ── Step 8: Test — verify CUDA is working ──
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
# Expected output:
#   2.7.0+cu128 | CUDA: True | GPU: NVIDIA GeForce RTX 5080 Laptop GPU

# ── Step 9: Deactivate venv ──
deactivate
.\.venv\Scripts\Activate.ps1
# ── Step 10: Install frontend dependencies ──
cd frontend
npm install
cd ..
```

**CUDA version mapping (Step 6):**

| Your GPU | Check with `nvidia-smi` | Use this tag |
|---|---|---|
| RTX 50xx (Blackwell) | CUDA Version: 12.8+ | `cu128` |
| RTX 40xx / 30xx | CUDA Version: 12.4 | `cu124` |
| RTX 30xx / 20xx | CUDA Version: 12.1 | `cu121` |
| Older | CUDA Version: 11.8 | `cu118` |

---

## First Run Setup

Copy the environment file and set a secret key:

```powershell
copy backend\.env.example backend\.env
```

Edit `backend\.env` and set `JWT_SECRET` to any random string (e.g. `mysecretkey123`).

---

## Project Structure

```
LLM_EdgeCraft/
├── frontend/          React 18 + TypeScript + Vite (port 5174)
├── backend/           FastAPI + SQLite + Unsloth (port 8001)
│   ├── .venv/         Python virtual environment (created by install.ps1)
│   ├── requirements.txt
│   └── run.py         Entry point — starts uvicorn server
├── scripts/
│   ├── install.ps1    Windows automated setup
│   ├── install.sh     Linux/Mac automated setup
│   ├── start-dev.ps1  Windows start both servers
│   └── start-dev.sh   Linux/Mac start both servers
├── training_config.yaml   Sample training config (upload in Hyperparameters page)
└── CLAUDE.md          Full developer reference for Claude Code sessions
```

---

## Storage Paths

All user data is stored in `C:\Users\{you}\.unslothcraft\`:

| Path | Contents |
|---|---|
| `~\.unslothcraft\outputs\` | LoRA adapter files from training |
| `~\.unslothcraft\exports\` | Exported GGUF / merged models |
| `~\.unslothcraft\assets\datasets\uploads\` | Uploaded datasets |
| `~\.unslothcraft\unslothcraft.db` | User accounts (SQLite) |

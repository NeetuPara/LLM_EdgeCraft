# CLAUDE.md — LLM EdgeCraft

Developer reference for Claude Code sessions. Read this before any changes.
**Single source of truth — ignore any CLAUDE.md files inside frontend/ or backend/.**

---

## Project Overview

**LLM EdgeCraft** is a full-stack LLM fine-tuning Studio:
- Select a base model (Llama, Qwen, Gemma, Mistral, etc.)
- Configure a dataset (HuggingFace or local upload)
- Launch fine-tuning on local GPU with live metric charts
- Chat with fine-tuned model (compare base vs fine-tuned side by side)
- Export as GGUF / merged 16-bit / LoRA adapter

**Design:** EdgeCraft glassmorphism (dark navy, cap-cyan, glass cards)
**ML engine:** Unsloth pip package (2-5x faster via Triton kernels)
**Auth:** Ported from EdgeCraft (email/password, PBKDF2, JWT HS256)

---

## Directory Structure

```
LLM_EdgeCraft/
├── frontend/          React 18 + TypeScript + Vite (port 5174)
├── backend/           FastAPI + SQLite + Unsloth (port 8001)
├── scripts/
│   ├── install.ps1    Windows: full automated setup
│   ├── install.sh     Linux/Mac: full automated setup
│   ├── start-dev.ps1  Windows: start both servers
│   └── start-dev.sh   Linux/Mac: start both servers
├── CLAUDE.md          this file
└── README.md
```

**Related (D:/EdgeCraft/ — NOT imported at runtime):**
- `unsloth-main/` — Unsloth source (reference only, installed as pip package)
- `Edge_ML_Platform-dev/` — EdgeCraft UI design reference

---

## Quick Start

```powershell
# First time only
.\scripts\install.ps1

# Every day — one command starts everything
.\scripts\start-dev.ps1        # demo mode (no backend)
.\scripts\start-dev.ps1 --real # real mode (GPU training)
```

The scripts auto-detect `backend/.venv` — no need to specify Python paths.

**To run backend manually from the backend/ directory:**
```powershell
cd backend

# Option 1: activate venv first (then just 'python')
.\.venv\Scripts\activate
python run.py

# Option 2: use venv Python directly (no activation needed)
.\.venv\Scripts\python.exe run.py
```

**Python venv after install:** `backend/.venv/` (Windows: `Scripts/python.exe`, Linux: `bin/python`)
The `.venv` has: unsloth, torch+CUDA, transformers, trl, peft, fastapi, sqlalchemy, etc.

---

## Frontend (`frontend/`)

### Tech Stack
React 18 · TypeScript · Vite 5 · Tailwind CSS v3 · Zustand v5 · React Router v6 ·
Framer Motion v11 · Recharts 2.x · `@xyflow/react` v12 · Dexie v4 · `@huggingface/hub` ·
Lucide React · Sonner (toasts) · dexie-react-hooks

### Running Modes
```bash
npm run dev        # demo mode (default) — reads .env.demo
npm run dev:demo   # VITE_DEMO_MODE=true  — mock data, no backend needed
npm run dev:real   # VITE_DEMO_MODE=false — real backend at :8001
npm run build      # production build (real mode)
```
`isMockMode()` checks `import.meta.env.VITE_DEMO_MODE === 'true'` — every API call routes to mock or real.

### Key Files
```
frontend/src/
├── App.tsx                   # BrowserRouter + AnimatePresence(mode=sync,initial=false)
│                             # + ErrorBoundary + all screens lazy-loaded via React.lazy()
├── main.tsx                  # Entry — NO <StrictMode> (breaks Framer Motion -> black screen)
├── index.css                 # All glass CSS classes + Tailwind directives
│
├── api/
│   ├── client.ts             # apiFetch(): JWT Bearer inject, 401 auto-refresh,
│   │                         # fires "auth:session-expired" CustomEvent on second 401
│   ├── auth-api.ts           # login(email,pw) signup(email,pw,name) me() logout()
│   │                         # IMPORTANT: uses EMAIL not username
│   ├── training-api.ts       # start/stop/reset/status/metrics/listRuns/deleteRun
│   │                         # TrainingStartRequest interface — field names must match backend
│   └── mock/
│       ├── index.ts          # isMockMode() + mockAuth/Training/Models/Datasets/System
│       └── data.ts           # MOCK_RUNS(4) MOCK_MODELS(10) MOCK_LORAS(2) MOCK_HARDWARE
│                             # MOCK_TRAINING_STATUS(400 total steps, starts at 0)
│
├── stores/
│   ├── auth-store.ts         # Zustand persist KEY:"unslothcraft-auth"
│   │                         # {user{id,email,name,role}, accessToken, isAuthenticated}
│   ├── training-config-store.ts  # Zustand persist KEY:"unslothcraft-training-config-v1" v:1
│   │                             # All wizard state: modelName,method,dataset,ALL hyperparams
│   ├── training-runtime-store.ts # EPHEMERAL (no persist)
│   │                             # lossHistory[] lrHistory[] logLines[] phase step
│   └── chat-store.ts         # Zustand persist (partial): params, loadedModel, compareMode
│
├── config/constants.ts       # PIPELINE_STAGES, GGUF_QUANT_OPTIONS, STATUS_COLORS
├── types/index.ts            # TrainingConfig interface + DEFAULT_TRAINING_CONFIG
│                             # Defaults: loraR=32,loraAlpha=32,lrScheduler=cosine,
│                             #   useRslora=true,trainOnCompletions=true,numEpochs=1,
│                             #   weightDecay=0.01, batchSize=2, gradAccumSteps=4
│
├── components/
│   ├── AnimatedBackground.tsx  # Canvas: blobs+constellation+particles (EdgeCraft port)
│   ├── NavBar.tsx             # Glass nav. Links: Dashboard/Chat/Export + pipeline pills
│   │                          # Pills shown on /new/* and /training routes
│   ├── Logo.tsx               # SVG hex + "UnslothCraft" gradient wordmark
│   ├── ProtectedRoute.tsx     # Redirects to /login if !isAuthenticated
│   └── InfoTooltip.tsx        # Hover tooltip
│
├── screens/
│   ├── LoginScreen.tsx        # Email+password + Create Account tabs
│   │                          # Demo: shows "Enter Demo" cyan button
│   ├── DashboardScreen.tsx    # Runs list + GPU VRAM bar + stats + Chat/Export quick actions
│   ├── TrainingScreen.tsx     # 4 charts (loss/eval/LR/gradNorm) + log console + progress
│   │                          # Y axes labeled: "Loss (lower=better)" / "Step size" / etc.
│   ├── ChatScreen.tsx         # Thread sidebar (Dexie) + model selector + SSE streaming
│   │                          # Compare mode: split panel, shared input, base vs LoRA
│   ├── ExportScreen.tsx       # Source+checkpoint picker + method cards + quant levels
│   ├── DataRecipesScreen.tsx  # HIDDEN from nav/dashboard — backend not implemented
│   ├── RecipeEditorScreen.tsx # HIDDEN — ReactFlow canvas editor
│   └── wizard/
│       ├── WizardShell.tsx              # Step N/4 header + content + Back/Next footer
│       ├── ModelSelectionScreen.tsx     # Step 1: Text/Vision selector + model search
│       │                               # Audio/Embeddings HIDDEN. Full Fine-tune HIDDEN.
│       ├── DatasetScreen.tsx            # Step 2: HF search + local upload + format + preview
│       ├── HyperparamsScreen.tsx        # Step 3: Essential + Advanced + RSLoRA/TrainOnComp
│       └── TrainingSummaryScreen.tsx    # Step 4: review + estimates + "Start Training"
│
└── hooks/
    ├── use-training-simulator.ts    # DEMO: 400 steps, 1 epoch, ~5 min, setInterval 1.5s
    ├── use-real-training-poller.ts  # REAL: polls GET /api/train/status every 1.5s
    │                               # Deduplicates — only appends log on new step
    └── use-demo-stream.ts          # DEMO: char-by-char LLM response simulation
```

### Design System

Tailwind `cap.*` colors:
```
cap-navy:    #0D1B2A   body background
cap-surface: #111827   card base
cap-panel:   #1E293B   panels/sidebars
cap-cyan:    #00A5D9   primary accent
cap-blue:    #0070AD   buttons/CTA
cap-border:  #334155   borders
cap-text:    #E2E8F0   primary text
cap-muted:   #94A3B8   secondary text
```

CSS classes (`index.css`):
```
.glass-panel          bg-slate-900/40 backdrop-blur-xl border-white/10 rounded-2xl
.glass-card           bg-slate-800/40 backdrop-blur-md border-white/10 rounded-xl p-6
.glass-card-interactive  + hover:border-cap-cyan/30 + hover:shadow-cap-cyan/10
.btn-primary          bg-[#0070AD] text-white hover:bg-[#0088CC]
.btn-secondary        bg-white/5 border-white/10 hover:bg-white/10
.btn-danger           bg-red-500/20 border-red-500/30 text-red-400
.glass-input          bg-slate-800/50 border-slate-700 focus:ring-cap-cyan/50
.skeleton             bg-slate-800/60 animate-pulse
```

### Critical Rules

**1. No StrictMode in main.tsx**
StrictMode double-mounts → Framer Motion animations cancelled → BLACK SCREEN.
Never add `<StrictMode>` back.

**2. PageTransition opacity must start at 1**
```tsx
initial={{ opacity: 1, y: 8 }}  // CORRECT
initial={{ opacity: 0, y: 8 }}  // WRONG — blank screen if animation delayed
```
All screens use `opacity: 1` in initial state.

**3. Lazy loading for all screens**
All screens use `React.lazy()` in App.tsx. If one screen crashes on import,
only that screen fails — not the whole app.

### Hidden Features (uncomment to re-enable)

| Feature | File | Location |
|---|---|---|
| Data Recipes nav | `components/NavBar.tsx` | NAV_LINKS array |
| Data Recipes dashboard | `screens/DashboardScreen.tsx` | quick actions array |
| Audio model type | `screens/wizard/ModelSelectionScreen.tsx` | MODEL_TYPES array |
| Embeddings model type | same | same |
| Full Fine-tune method | same | METHODS array |
| Experiment Tracking (WandB/TensorBoard) | `screens/wizard/HyperparamsScreen.tsx` | Section block |

---

## Backend (`backend/`)

### Tech Stack
FastAPI 0.115+ · Uvicorn · SQLite (raw sqlite3 for training, SQLAlchemy for users) ·
Python-Jose JWT HS256 · PBKDF2-HMAC-SHA256 (stdlib) · Unsloth pip package

### CRITICAL RULE
Never import from `unsloth-main/studio/backend/` at runtime.
All code is self-contained or uses `from unsloth import FastLanguageModel` (pip).

### Key Files
```
backend/
├── main.py              # FastAPI, CORS(*), lifespan: init_db()+detect_hardware()
├── run.py               # Uvicorn :8001, access_log=False, log_level=warning
│                        # (Suppresses HTTP 200 spam from 1.5s polling)
│
├── config/settings.py   # JWT_SECRET, DATABASE_URL(SQLite ~/.unslothcraft/),
│                        # CORS_ORIGINS, PORT=8001
│
├── auth/
│   ├── routes.py        # signup/login/logout/me/change-password
│   ├── jwt_handler.py   # HS256 1-day: create_token({user_id,role}) / decode_token()
│   ├── hashing.py       # PBKDF2 stdlib. Stored as "pbkdf2$salt_hex$hash_hex"
│   └── dependencies.py  # get_current_user(): Bearer header first, cookie fallback
│
├── database/
│   ├── database.py      # SQLite, check_same_thread=False, get_db(), init_db()
│   └── models.py        # User: id,email(unique),name,password_hash,role,created_at
│
├── storage/studio_db.py # Raw sqlite3, WAL mode, foreign_keys=ON
│                        # Tables: training_runs, training_metrics(UNIQUE run_id+step),
│                        #         scan_folders
│
├── core/training/
│   ├── training.py      # TrainingBackend: mp.Queue subprocess + pump thread
│   │                    # _pump_loop() updates histories + flushes SQLite every 10 events
│   └── worker.py        # run_training_process(): SELF-CONTAINED
│                        # from_pretrained -> get_peft_model -> load_dataset ->
│                        # _format_dataset_for_training -> SFTTrainer -> save_pretrained
│
├── core/inference/
│   └── manager.py       # GGUF: llama-server subprocess + httpx proxy to /v1/chat/completions
│                        # HF models: NOT YET IMPLEMENTED (returns error)
│
├── core/export/
│   ├── orchestrator.py  # Persistent subprocess, cmd/resp queues, 3600s timeout
│   └── worker.py        # model.save_pretrained_gguf/merged/lora via unsloth
│
├── routers/             # See API Endpoints below
│
├── utils/hardware/hardware.py    # detect_hardware(), DEVICE global, get_gpu_summary()
├── utils/hardware/nvidia.py      # nvidia-smi: get_primary_gpu_utilization()
├── utils/models/checkpoints.py   # scan_checkpoints(), scan_trained_loras()
├── utils/paths/storage_roots.py  # All ~/.unslothcraft/* paths
└── utils/datasets/format_checker.py  # Self-contained format detection
```

**`unsloth_compiled_cache/`** — Auto-generated Triton kernels. Safe to delete. Do not commit.

### API Endpoints

```
Auth (POST uses email not username):
  POST /api/auth/signup|login|logout|change-password   GET /api/auth/me

System:
  GET /api/health                    no auth, returns device_type
  GET /api/system/hardware           GPU name, VRAM, package versions
  GET /api/train/hardware            live GPU utilization (polled during training)

Training:
  POST /api/train/start              body: TrainingStartRequest
  POST /api/train/stop               body: {save:bool}
  POST /api/train/reset
  GET  /api/train/status             includes metric_history for chart recovery on reconnect
  GET  /api/train/metrics
  GET  /api/train/progress           SSE stream
  GET  /api/train/runs               paginated history
  GET  /api/train/runs/{id}          run + metrics
  DELETE /api/train/runs/{id}

Models:
  GET /api/models/list               HF cached + defaults
  GET /api/models/local              scan HF cache + custom folders
  GET /api/models/loras              scan ~/.unslothcraft/outputs/
  GET /api/models/checkpoints        scan with loss info
  GET /api/models/config/{name}      capabilities, recommended params
  GET/POST/DELETE /api/models/scan-folders

Inference:
  POST /api/inference/load           GGUF only (HF not implemented yet)
  POST /api/inference/unload
  GET  /api/inference/status
  POST /v1/chat/completions          OpenAI SSE streaming
  GET  /v1/models

Export:
  POST /api/export/load-checkpoint
  GET  /api/export/status
  GET  /api/export/checkpoints
  POST /api/export/export/gguf|merged|lora|base
  POST /api/export/cleanup

Datasets:
  POST /api/datasets/check-format
  POST /api/datasets/upload
  GET  /api/datasets/local
  POST /api/datasets/ai-assist-mapping
```

### Training Field Names (common bugs)
```python
# CORRECT backend field names:
hf_dataset      (NOT "dataset")
local_datasets  (NOT "local_dataset" — must be plural list)
training_type   "LoRA/QLoRA" for QLoRA/LoRA, "full" for full finetune
use_lora        True for LoRA/QLoRA, False for full
```

### Training Pipeline

```
1. FastLanguageModel.from_pretrained(model_name, load_in_4bit=True)
   Downloads to ~/.cache/huggingface/hub/ if not cached
   Unsloth auto-maps to pre-quantized 4-bit version

2. FastLanguageModel.get_peft_model(model, r=32, lora_alpha=32, use_rslora=True)
   Attaches LoRA to q/k/v/o/gate/up/down projections

3. load_dataset(hf_dataset, split="train")

4. _format_dataset_for_training(dataset, tokenizer, model_name)
   CRITICAL: converts any format to model's specific chat template
   alpaca: {instruction,output} -> messages -> tokenizer.apply_chat_template()
   sharegpt: {conversations:[{from:human/gpt}]} -> normalize -> chat template
   Output: single "text" column with model-specific tokens
   Llama3: <|start_header_id|>user<|end_header_id|>...<|eot_id|>
   Qwen:   <|im_start|>user...<|im_end|>

5. SFTTrainer(..., args=SFTConfig(lr_scheduler_type="cosine", optim="adamw_8bit"))

6. model.save_pretrained(~/.unslothcraft/outputs/{model}_{timestamp}/)
   adapter_config.json + adapter_model.safetensors (~30-200MB)
```

### Dataset Format Detection

| Format | Detected by | Converted to |
|---|---|---|
| alpaca | `instruction`+`output` cols | messages list -> chat template |
| sharegpt | `conversations` col | normalize human/gpt roles -> chat template |
| chatml | `messages` col | normalize user/assistant -> chat template |
| qa | `question`+`answer` cols | messages -> chat template |
| completion | `prompt`+`completion` cols | messages -> chat template |
| text | `text` col already | used directly |

### Storage Paths

```
~/.unslothcraft/
├── unslothcraft.db       SQLite: users (SQLAlchemy)
├── outputs/              LoRA adapters from training
│   └── {model}_{ts}/adapter_config.json + adapter_model.safetensors
├── exports/              GGUF/merged from Export page
└── datasets/uploads/     user-uploaded datasets

~/.cache/huggingface/hub/ HF model cache (shared)
```

### Key Design Decisions

| Decision | Why |
|---|---|
| PBKDF2 not bcrypt | bcrypt 5.0 broke passlib. Stdlib = zero deps. |
| SQLite not PostgreSQL | Zero setup. Switch via DATABASE_URL env var. |
| access_log=False | Suppresses HTTP 200 spam (40 req/min from 1.5s polling) |
| Subprocess for training/export | Fresh spawn process = no stale torch state |
| llama-server for GGUF | Fastest inference, OpenAI-compat /v1/chat/completions |
| Polling not SSE | SSE needs fetch()+ReadableStream for auth. Polling simpler, 1.5s delay fine. |

---

## Common Issues

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: sqlalchemy` | Wrong Python. Activate venv: `cd backend && .\.venv\Scripts\activate` then `python run.py` |
| Black screen | F12 Console. ErrorBoundary shows error. Check opacity:1 on all motion.divs. |
| `null.toFixed()` crash | Use `val != null` not `val !== undefined` |
| ECONNREFUSED :8001 | Backend not running. `cd backend && python run.py` |
| `You must specify a formatting_func` | `_format_dataset_for_training()` not called before SFTTrainer |
| Training start does nothing | Check field names: `hf_dataset` not `dataset`, `local_datasets` not `local_dataset` |
| Charts empty in real mode | `useRealTrainingPoller` deduplicates — only logs on new step. Check training active. |
| Demo mode shows in real mode | `.env.real` must have `VITE_DEMO_MODE=false`. Run `npm run dev:real`. |
| `utils.transformers_version` error | Old worker loading unsloth-main. Worker must `from unsloth import FastLanguageModel`. |
| CUDA OOM | Reduce `batch_size` to 1 or `max_seq_length` to 1024 |

---

## Not Yet Implemented

| Feature | Notes |
|---|---|
| Data Recipes backend | Frontend UI exists (hidden). Needs Gretel data_designer library. |
| SSE for training | Switch from polling to fetch()+ReadableStream (unsloth-main pattern) |
| HF model inference | Only GGUF works. inference/manager.py returns error for HF models. |
| Download progress bar | Monitor subprocess tqdm bars, forward via event_queue |
| Audio fine-tuning | Unsloth supports it. Needs wizard + worker changes. |
| Embeddings fine-tuning | FastSentenceTransformer + MultipleNegativesRankingLoss |
| Full Fine-tune | Uncomment METHODS + test. worker.py handles full_finetuning=True already. |
| Experiment tracking | WandB/TensorBoard hidden. Backend worker already supports both. |

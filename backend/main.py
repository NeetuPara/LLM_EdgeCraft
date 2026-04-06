import sys
import os
from pathlib import Path

_backend_dir = str(Path(__file__).parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import CORS_ORIGINS
from database.database import init_db
from auth.routes import router as auth_router
from routers.system import router as system_router
from routers.models import router as models_router
from routers.training import router as training_router
from routers.training_history import router as training_history_router
from routers.inference import inference_router, openai_router
from routers.export import router as export_router
from routers.datasets import router as datasets_router
from routers.config_parser import router as config_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    init_db()

    try:
        from utils.paths import ensure_base_directories
        ensure_base_directories()
    except Exception:
        base = Path.home() / ".unslothcraft"
        for subdir in ["outputs", "exports", "datasets/uploads"]:
            (base / subdir).mkdir(parents=True, exist_ok=True)

    try:
        from utils.hardware.hardware import detect_hardware
        detect_hardware()
    except Exception as e:
        print(f"Hardware detection skipped: {e}")

    try:
        from storage.studio_db import cleanup_orphaned_runs
        cleanup_orphaned_runs()
    except Exception:
        pass

    yield
    # ── Shutdown ──
    try:
        from core.training import get_training_backend
        backend = get_training_backend()
        if backend.is_training_active():
            backend.force_terminate()
    except Exception:
        pass
    try:
        from core.inference import get_inference_manager
        get_inference_manager().unload()
    except Exception:
        pass
    try:
        from core.export import get_export_backend
        get_export_backend()._shutdown_subprocess(timeout=5)
    except Exception:
        pass


app = FastAPI(
    title="UnslothCraft Backend",
    version="0.3.0",
    description="Custom backend: EdgeCraft auth + Unsloth ML",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)

# ── Routers ──
app.include_router(auth_router, prefix="/api")
app.include_router(system_router)
app.include_router(models_router)
app.include_router(training_router)
app.include_router(training_history_router)
app.include_router(inference_router)
app.include_router(openai_router)
app.include_router(export_router)
app.include_router(datasets_router)
app.include_router(config_router)

# ── 501 stub for phases 4+ ──
from fastapi import APIRouter
from fastapi.responses import JSONResponse

_IMPLEMENTED_PREFIXES = [
    "/api/auth", "/api/health", "/api/system",
    "/api/models", "/api/train", "/api/inference",
    "/api/export", "/api/datasets", "/api/config",
    "/api/models/vram-estimate", "/v1",
]

stub = APIRouter()

@stub.api_route(
    "/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    include_in_schema=False,
)
async def catch_all(path: str):
    full = f"/api/{path}"
    for prefix in _IMPLEMENTED_PREFIXES:
        if full.startswith(prefix):
            return JSONResponse({"detail": "Not found"}, status_code=404)
    return JSONResponse(
        {"detail": f"/{path} not yet implemented — coming in Phase 4+"},
        status_code=501,
    )

app.include_router(stub)

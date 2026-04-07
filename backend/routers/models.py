"""
/api/models/* endpoints.
Phase 2: model listing, local scans, checkpoints, scan folders.
"""
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List

from auth.dependencies import get_current_user
from storage.studio_db import list_scan_folders, add_scan_folder, remove_scan_folder
from utils.models.checkpoints import scan_checkpoints, scan_trained_loras

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])

# ── Curated models ≤4B, full precision only (no -bnb-4bit, no GGUF) ──
# type field drives text/vision tab split in frontend
_DEFAULT_MODELS = [
    # ── Text ──
    {"id": "m1", "name": "unsloth/Llama-3.2-1B-Instruct",          "params": "1B",   "size_gb": 0.8, "org": "Meta",        "year": 2024, "is_gated": True,  "type": "text",   "is_local": False},
    {"id": "m2", "name": "unsloth/Llama-3.2-3B-Instruct",          "params": "3B",   "size_gb": 2.0, "org": "Meta",        "year": 2024, "is_gated": True,  "type": "text",   "is_local": False},
    {"id": "m3", "name": "unsloth/gemma-3-1b-it",                  "params": "1B",   "size_gb": 0.7, "org": "Google",      "year": 2025, "is_gated": True,  "type": "text",   "is_local": False},
    {"id": "m3v","name": "unsloth/gemma-3-4b-it",                  "params": "4B",   "size_gb": 2.5, "org": "Google",      "year": 2025, "is_gated": True,  "type": "text",   "is_local": False},
    {"id": "m4", "name": "unsloth/Qwen3-1.7B",                     "params": "1.7B", "size_gb": 1.1, "org": "Qwen",        "year": 2025, "is_gated": False, "type": "text",   "is_local": False},
    {"id": "m5", "name": "unsloth/Qwen2.5-3B-Instruct",            "params": "3B",   "size_gb": 2.0, "org": "Qwen",        "year": 2024, "is_gated": False, "type": "text",   "is_local": False},
    {"id": "m6", "name": "unsloth/Phi-4-mini-Instruct",            "params": "3.8B", "size_gb": 2.4, "org": "Microsoft",   "year": 2025, "is_gated": False, "type": "text",   "is_local": False},
    {"id": "m7", "name": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B", "params": "1.5B", "size_gb": 1.0, "org": "DeepSeek",    "year": 2025, "is_gated": False, "type": "text",   "is_local": False},
    # ── Vision ──
    {"id": "v1", "name": "unsloth/gemma-3-4b-it",                  "params": "4B",   "size_gb": 2.5, "org": "Google",      "year": 2025, "is_gated": True,  "type": "vision", "is_local": False},
    {"id": "v2", "name": "unsloth/Qwen2.5-VL-3B-Instruct",         "params": "3B",   "size_gb": 2.0, "org": "Qwen",        "year": 2024, "is_gated": False, "type": "vision", "is_local": False},
    {"id": "v3", "name": "HuggingFaceTB/SmolVLM-500M-Instruct",    "params": "500M", "size_gb": 0.4, "org": "HuggingFace", "year": 2024, "is_gated": False, "type": "vision", "is_local": False},
    {"id": "v4", "name": "HuggingFaceTB/SmolVLM-500M-Base",        "params": "500M", "size_gb": 0.4, "org": "HuggingFace", "year": 2024, "is_gated": False, "type": "vision", "is_local": False},
]


def _hf_cache_dirs() -> list[Path]:
    """Collect all HuggingFace cache directories to scan."""
    candidates = [
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "hub")
    hf_hub = os.environ.get("HF_HUB_CACHE")
    if hf_hub:
        candidates.append(Path(hf_hub))
    return [p for p in candidates if p.exists()]


def _scan_hf_cache() -> list[dict]:
    """Scan HF cache for model directories."""
    models = []
    seen: set[str] = set()

    for cache_dir in _hf_cache_dirs():
        try:
            for item in cache_dir.iterdir():
                if not item.is_dir() or not item.name.startswith("models--"):
                    continue
                # Convert models--org--name → org/name
                model_id = item.name[len("models--"):].replace("--", "/", 1)
                if model_id in seen:
                    continue
                seen.add(model_id)
                # Estimate size
                try:
                    snapshots = item / "snapshots"
                    size_bytes = 0
                    if snapshots.exists():
                        for snap in snapshots.iterdir():
                            if snap.is_dir():
                                for f in snap.rglob("*.safetensors"):
                                    size_bytes += f.stat().st_size
                                for f in snap.rglob("*.bin"):
                                    size_bytes += f.stat().st_size
                    size_gb = round(size_bytes / 1024 ** 3, 2) if size_bytes else None
                except Exception:
                    size_gb = None

                models.append({
                    "id": model_id,
                    "name": model_id,
                    "size_gb": size_gb,
                    "is_local": True,
                    "path": str(item),
                })
        except Exception as e:
            logger.debug("Error scanning %s: %s", cache_dir, e)

    return models


def _scan_directory_for_models(directory: str) -> list[dict]:
    """Scan a directory for model folders (adapter_config.json or config.json)."""
    models = []
    try:
        p = Path(directory)
        if not p.exists() or not p.is_dir():
            return models
        for item in p.iterdir():
            if not item.is_dir():
                continue
            if (item / "config.json").exists() or (item / "adapter_config.json").exists():
                models.append({
                    "id": str(item),
                    "name": item.name,
                    "size_gb": None,
                    "is_local": True,
                    "path": str(item),
                })
    except Exception as e:
        logger.debug("Error scanning %s: %s", directory, e)
    return models


# ── Routes ──

@router.get("/list")
def list_models(_user=Depends(get_current_user)):
    """List available models: HF cached + default unsloth models."""
    local = _scan_hf_cache()
    local_ids = {m["name"] for m in local}
    # Merge with defaults, marking which are cached locally
    result = list(local)
    cache_names = set(local_ids)
    seen_typed: set[str] = set()   # tracks "name|type" for typed defaults
    for m in _DEFAULT_MODELS:
        mtype = m.get("type", "")
        key = f"{m['name']}|{mtype}"
        if key not in seen_typed:
            seen_typed.add(key)
            # Typed defaults always included (frontend tab filter handles routing).
            # Untyped defaults skipped if HF cache already has the model.
            if mtype or m["name"] not in cache_names:
                result.append(m)
    return result


@router.get("/local")
def list_local_models(_user=Depends(get_current_user)):
    """List models found in HF cache + custom scan folders."""
    models = _scan_hf_cache()
    seen = {m["name"] for m in models}

    # Scan custom scan folders
    for folder in list_scan_folders():
        for m in _scan_directory_for_models(folder["path"]):
            if m["name"] not in seen:
                seen.add(m["name"])
                models.append(m)

    return models


@router.get("/loras")
def list_loras(_user=Depends(get_current_user)):
    """Scan outputs directory for trained LoRA adapters."""
    loras = scan_trained_loras()
    return [
        {"id": path, "name": name, "path": path, "base_model": None}
        for name, path in loras
    ]


@router.get("/checkpoints")
def list_checkpoints(_user=Depends(get_current_user)):
    """Scan outputs directory for training checkpoints."""
    runs = scan_checkpoints()
    result = []
    for model_name, checkpoints, metadata in runs:
        for display_name, path, loss in checkpoints:
            result.append({
                "id": path,
                "path": path,
                "step": None,
                "loss": loss,
                "run_id": model_name,
                "base_model": metadata.get("base_model"),
                "is_final": display_name == model_name,
                "display_name": display_name,
            })
    return result


@router.get("/config/{model_name:path}")
def get_model_config(model_name: str, _user=Depends(get_current_user)):
    """Get capabilities and defaults for a model."""
    lower = model_name.lower()

    is_vision = any(x in lower for x in ["vl", "vision", "llava", "mllama", "pixtral"])
    is_audio = any(x in lower for x in ["whisper", "audio", "speech", "orpheus"])
    is_embedding = any(x in lower for x in ["embedding", "e5", "bge", "sentence"])

    # Rough parameter count from name
    max_pos = 8192
    if "1b" in lower or "1.5b" in lower:
        max_pos = 32768
    elif "3b" in lower or "4b" in lower:
        max_pos = 131072
    elif "7b" in lower or "8b" in lower:
        max_pos = 131072
    elif "13b" in lower or "14b" in lower:
        max_pos = 131072

    return {
        "name": model_name,
        "is_vision": is_vision,
        "is_audio": is_audio,
        "is_embedding": is_embedding,
        "max_position_embeddings": max_pos,
        "recommended_lora_r": 64,
        "recommended_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "trust_remote_code": "qwen" in lower or "glm" in lower,
    }


# ── Scan folder management ──

class ScanFolderReq(BaseModel):
    path: str


@router.get("/scan-folders")
def get_scan_folders(_user=Depends(get_current_user)):
    return list_scan_folders()


@router.post("/scan-folders")
def create_scan_folder(body: ScanFolderReq, _user=Depends(get_current_user)):
    try:
        return add_scan_folder(body.path)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/scan-folders/{folder_id}")
def delete_scan_folder(folder_id: int, _user=Depends(get_current_user)):
    remove_scan_folder(folder_id)
    return {"msg": "Removed"}


# ── Cached model management ──

@router.get("/exported")
def list_exported_models(_user=Depends(get_current_user)):
    """Scan exports folder for merged models and LoRA adapters."""
    from utils.paths import exports_root, outputs_root
    models = []
    for scan_dir, model_type in [(exports_root(), "merged"), (outputs_root(), "lora")]:
        if not scan_dir.exists():
            continue
        for item in sorted(scan_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not item.is_dir():
                continue
            is_lora = (item / "adapter_config.json").exists()
            is_merged = (item / "config.json").exists() and not is_lora
            if is_lora or is_merged:
                models.append({
                    "id": str(item),
                    "name": item.name,
                    "path": str(item),
                    "type": "lora" if is_lora else "merged",
                    "size_gb": None,
                })
    return models


@router.get("/cached-models")
def list_cached_models(_user=Depends(get_current_user)):
    """List all non-GGUF cached models."""
    return _scan_hf_cache()


@router.get("/cached-gguf")
def list_cached_gguf(_user=Depends(get_current_user)):
    """List cached GGUF repos from HF cache."""
    gguf_models = []
    for cache_dir in _hf_cache_dirs():
        try:
            for item in cache_dir.iterdir():
                if not item.is_dir() or not item.name.startswith("models--"):
                    continue
                # Check if any snapshot contains .gguf files
                snapshots = item / "snapshots"
                if not snapshots.exists():
                    continue
                gguf_files = list(snapshots.rglob("*.gguf"))
                if not gguf_files:
                    continue
                model_id = item.name[len("models--"):].replace("--", "/", 1)
                variants = [f.name for f in gguf_files]
                gguf_models.append({
                    "repo_id": model_id,
                    "path": str(item),
                    "variants": variants,
                    "count": len(variants),
                })
        except Exception:
            pass
    return gguf_models


# ── VRAM Estimation ──

class VramEstimateRequest(BaseModel):
    model_name: str
    training_method: str = "qlora"
    lora_rank: int = 32
    target_modules: List[str] = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    batch_size: int = 2
    max_seq_length: int = 2048
    optimizer: str = "adamw_8bit"
    gradient_checkpointing: str = "unsloth"
    load_in_4bit: bool = True
    hf_token: Optional[str] = None


def _build_cfg_obj(raw: dict):
    """Convert a raw config dict to a namespace object.
    Also nests text_config and vision_config as sub-objects."""
    class _Cfg:
        pass

    def _to_obj(d: dict):
        o = _Cfg()
        for k, v in d.items():
            setattr(o, k, v)
        return o

    obj = _to_obj(raw)
    if hasattr(obj, "text_config") and isinstance(obj.text_config, dict):
        obj.text_config = _to_obj(obj.text_config)
    if hasattr(obj, "vision_config") and isinstance(obj.vision_config, dict):
        obj.vision_config = _to_obj(obj.vision_config)
    return obj


def _fetch_model_arch(model_name: str, hf_token: Optional[str]):
    """Fetch model config.json and extract architecture params + optional vision_config.
    Returns (arch, vision_config) tuple. vision_config is None for text-only models."""
    import json
    from utils.hardware.vram_estimation import extract_arch_config

    def _parse(raw: dict):
        obj = _build_cfg_obj(raw)
        arch = extract_arch_config(obj)
        vision_cfg = getattr(obj, "vision_config", None)
        return arch, vision_cfg

    # 1. Try HF cache — find config.json in snapshots
    for cache_dir in _hf_cache_dirs():
        cache_key = "models--" + model_name.replace("/", "--")
        model_dir = cache_dir / cache_key
        if model_dir.exists():
            snaps_dir = model_dir / "snapshots"
            for snap in (snaps_dir.iterdir() if snaps_dir.exists() else []):
                cfg_path = snap / "config.json"
                if cfg_path.exists():
                    try:
                        arch, vision_cfg = _parse(json.loads(cfg_path.read_text()))
                        if arch:
                            return arch, vision_cfg
                    except Exception:
                        pass

    # 2. Download config.json only via huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            token=hf_token or None,
        )
        arch, vision_cfg = _parse(json.loads(Path(cfg_path).read_text()))
        if arch:
            return arch, vision_cfg
    except Exception as e:
        logger.debug("hf_hub_download config failed: %s", e)

    return None, None


@router.post("/vram-estimate")
def vram_estimate(body: VramEstimateRequest, _user=Depends(get_current_user)):
    """
    Compute accurate training VRAM estimate using real model architecture.
    Returns per-component breakdown in GB.
    """
    from utils.hardware.vram_estimation import (
        TrainingVramConfig, estimate_training_vram,
    )

    arch, vision_cfg = _fetch_model_arch(body.model_name, body.hf_token)
    if arch is None:
        raise HTTPException(
            422,
            f"Could not fetch architecture config for '{body.model_name}'. "
            "Make sure the model is cached locally or accessible on HuggingFace."
        )

    train_cfg = TrainingVramConfig(
        training_method=body.training_method,
        batch_size=body.batch_size,
        max_seq_length=body.max_seq_length,
        lora_rank=body.lora_rank,
        target_modules=body.target_modules,
        gradient_checkpointing=body.gradient_checkpointing,
        optimizer=body.optimizer,
        load_in_4bit=body.load_in_4bit,
    )

    breakdown = estimate_training_vram(arch, train_cfg, vision_config=vision_cfg)
    return {
        "total_gb": round(breakdown.total / 1024**3, 2),
        "breakdown": breakdown.to_gb_dict(),
        "model_name": body.model_name,
        "is_vlm": vision_cfg is not None,
    }

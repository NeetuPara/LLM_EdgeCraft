"""
Export routes — Phase 5.
Adapted from unsloth-main/studio/backend/routes/export.py
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/export", tags=["export"])


# ── Request models ──

class LoadCheckpointRequest(BaseModel):
    checkpoint_path: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    trust_remote_code: bool = False
    hf_token: Optional[str] = None


class ExportMergedRequest(BaseModel):
    save_directory: str = ""
    format_type: str = "16-bit (FP16)"
    push_to_hub: bool = False
    repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    private: bool = False


class ExportBaseRequest(BaseModel):
    save_directory: str = ""
    push_to_hub: bool = False
    repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    private: bool = False
    base_model_id: Optional[str] = None


class ExportGGUFRequest(BaseModel):
    save_directory: str = ""
    quantization_method: str = "Q4_K_M"
    push_to_hub: bool = False
    repo_id: Optional[str] = None
    hf_token: Optional[str] = None


class ExportLoRARequest(BaseModel):
    save_directory: str = ""
    push_to_hub: bool = False
    repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    private: bool = False


def _resolve_save_dir(save_directory: str, export_type: str) -> str:
    """Resolve save directory — absolute or relative under exports root."""
    if not save_directory or save_directory.strip() == "":
        from utils.paths import exports_root
        return str(exports_root() / export_type)
    p = Path(save_directory.strip())
    if p.is_absolute():
        return str(p)
    from utils.paths import resolve_export_dir
    return str(resolve_export_dir(save_directory))


# ── Routes ──

@router.post("/load-checkpoint")
async def load_checkpoint(request: LoadCheckpointRequest, _user=Depends(get_current_user)):
    """Load a checkpoint into the export subprocess. Required before any export."""
    # Free GPU memory from inference/training first
    try:
        from core.inference import get_inference_manager
        mgr = get_inference_manager()
        if mgr.is_loaded:
            logger.info("Unloading inference model to free GPU for export")
            mgr.unload()
    except Exception as e:
        logger.warning("Could not unload inference model: %s", e)

    try:
        from core.training import get_training_backend
        trn = get_training_backend()
        if trn.is_training_active():
            logger.info("Stopping training to free GPU for export")
            trn.stop_training()
    except Exception as e:
        logger.warning("Could not stop training: %s", e)

    try:
        from core.export import get_export_backend
        backend = get_export_backend()
        success, message = await asyncio.to_thread(
            backend.load_checkpoint,
            request.checkpoint_path,
            request.max_seq_length,
            request.load_in_4bit,
            request.trust_remote_code,
            request.hf_token,
        )
        if not success:
            raise HTTPException(400, message)
        return {"success": True, "message": message,
                "is_vision": backend.is_vision, "is_peft": backend.is_peft,
                "checkpoint": backend.current_checkpoint}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error loading checkpoint: %s", e, exc_info=True)
        raise HTTPException(500, f"Failed to load checkpoint: {e}")


@router.get("/status")
def get_export_status(_user=Depends(get_current_user)):
    """Get current export backend status."""
    from core.export import get_export_backend
    backend = get_export_backend()
    return {
        "loaded_checkpoint": backend.current_checkpoint,
        "is_vision": bool(backend.is_vision),
        "is_peft": bool(backend.is_peft),
        "subprocess_alive": backend._ensure_subprocess_alive(),
    }


@router.post("/cleanup")
async def cleanup_export(_user=Depends(get_current_user)):
    """Shut down export subprocess and free GPU memory."""
    from core.export import get_export_backend
    backend = get_export_backend()
    success = await asyncio.to_thread(backend.cleanup_memory)
    if not success:
        raise HTTPException(500, "Memory cleanup failed")
    return {"success": True, "message": "Export memory freed"}


@router.post("/export/merged")
async def export_merged(request: ExportMergedRequest, _user=Depends(get_current_user)):
    """Export merged PEFT model (16-bit or 4-bit)."""
    from core.export import get_export_backend
    backend = get_export_backend()
    save_dir = _resolve_save_dir(request.save_directory, "merged")
    try:
        success, message = await asyncio.to_thread(
            backend.export_merged_model,
            save_dir, request.format_type, request.push_to_hub,
            request.repo_id, request.hf_token, request.private,
        )
        if not success:
            raise HTTPException(400, message)
        return {"success": True, "message": message, "save_directory": save_dir}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error exporting merged model: %s", e, exc_info=True)
        raise HTTPException(500, f"Export failed: {e}")


@router.post("/export/base")
async def export_base(request: ExportBaseRequest, _user=Depends(get_current_user)):
    """Export non-PEFT base model."""
    from core.export import get_export_backend
    backend = get_export_backend()
    save_dir = _resolve_save_dir(request.save_directory, "base")
    try:
        success, message = await asyncio.to_thread(
            backend.export_base_model,
            save_dir, request.push_to_hub, request.repo_id,
            request.hf_token, request.private, request.base_model_id,
        )
        if not success:
            raise HTTPException(400, message)
        return {"success": True, "message": message, "save_directory": save_dir}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error exporting base model: %s", e, exc_info=True)
        raise HTTPException(500, f"Export failed: {e}")


@router.post("/export/gguf")
async def export_gguf(request: ExportGGUFRequest, _user=Depends(get_current_user)):
    """Export to GGUF format. Can take 30+ minutes for large models."""
    from core.export import get_export_backend
    backend = get_export_backend()
    save_dir = _resolve_save_dir(request.save_directory, "gguf")
    try:
        success, message = await asyncio.to_thread(
            backend.export_gguf,
            save_dir, request.quantization_method, request.push_to_hub,
            request.repo_id, request.hf_token,
        )
        if not success:
            raise HTTPException(400, message)
        return {"success": True, "message": message, "save_directory": save_dir,
                "quantization_method": request.quantization_method}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error exporting GGUF: %s", e, exc_info=True)
        raise HTTPException(500, f"Export failed: {e}")


@router.post("/export/lora")
async def export_lora(request: ExportLoRARequest, _user=Depends(get_current_user)):
    """Export LoRA adapter weights only."""
    from core.export import get_export_backend
    backend = get_export_backend()
    save_dir = _resolve_save_dir(request.save_directory, "lora")
    try:
        success, message = await asyncio.to_thread(
            backend.export_lora_adapter,
            save_dir, request.push_to_hub, request.repo_id,
            request.hf_token, request.private,
        )
        if not success:
            raise HTTPException(400, message)
        return {"success": True, "message": message, "save_directory": save_dir}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error exporting LoRA: %s", e, exc_info=True)
        raise HTTPException(500, f"Export failed: {e}")


@router.get("/checkpoints")
def list_checkpoints(_user=Depends(get_current_user)):
    """List available training checkpoints."""
    from core.export import get_export_backend
    backend = get_export_backend()
    runs = backend.scan_checkpoints()
    result = []
    for model_name, checkpoints, metadata in runs:
        for display_name, path, loss in checkpoints:
            result.append({
                "path": path,
                "display_name": display_name,
                "run_name": model_name,
                "loss": loss,
                "base_model": metadata.get("base_model"),
                "peft_type": metadata.get("peft_type"),
                "lora_rank": metadata.get("lora_rank"),
                "is_final": display_name == model_name,
            })
    return result

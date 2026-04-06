"""
System + hardware endpoints.
Phase 2: uses hardware detection module.
"""
import platform
import sys
import logging
from datetime import datetime
from fastapi import APIRouter, Depends
from auth.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


def _get_gpu_utilization() -> int | None:
    try:
        from utils.hardware.nvidia import get_primary_gpu_utilization
        result = get_primary_gpu_utilization()
        return result.get("gpu_utilization") if result.get("available") else None
    except Exception:
        return None


def _get_hardware_state_safe() -> dict:
    """Return device_type + chat_only, triggering detection if not yet run."""
    try:
        from utils.hardware.hardware import DEVICE, get_hardware_state, detect_hardware
        if DEVICE is None:
            detect_hardware()
        return get_hardware_state()
    except Exception:
        return {"device_type": "cpu", "chat_only": True}


@router.get("/api/health")
def health():
    """Basic health check — no auth required."""
    state = _get_hardware_state_safe()
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "platform": sys.platform,
        "device_type": state.get("device_type", "cpu"),
        "chat_only": state.get("chat_only", True),
    }


@router.get("/api/system")
def system_info(_user=Depends(get_current_user)):
    """System info — platform, CPU, memory."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        mem_total = round(mem.total / 1024 ** 3, 2)
        mem_available = round(mem.available / 1024 ** 3, 2)
        cpu_count = psutil.cpu_count(logical=True)
    except ImportError:
        mem_total = None
        mem_available = None
        cpu_count = None

    return {
        "platform": sys.platform,
        "python_version": sys.version,
        "cpu_count": cpu_count,
        "memory_total_gb": mem_total,
        "memory_available_gb": mem_available,
    }


@router.get("/api/system/hardware")
def hardware_info(_user=Depends(get_current_user)):
    """GPU info + ML package versions."""
    try:
        from utils.hardware.hardware import get_gpu_summary, get_package_versions
        summary = get_gpu_summary()
        versions = get_package_versions()
        return {
            "gpu_name":               summary.get("gpu_name"),
            "gpu_memory_total_gb":    summary.get("vram_total_gb"),
            "gpu_memory_free_gb":     summary.get("vram_free_gb"),
            "gpu_memory_used_gb":     summary.get("vram_used_gb"),
            "gpu_utilization":        _get_gpu_utilization(),
            "device":                 summary.get("device", "cpu"),
            "torch_version":          versions.get("torch"),
            "cuda_version":           versions.get("cuda"),
            "transformers_version":   versions.get("transformers"),
            "unsloth_version":        versions.get("unsloth"),
        }
    except Exception as e:
        logger.warning("hardware_info error: %s", e)
        return {
            "gpu_name": None, "gpu_memory_total_gb": None,
            "gpu_memory_free_gb": None, "gpu_memory_used_gb": None,
            "device": "cpu", "torch_version": None, "cuda_version": None,
            "transformers_version": None, "unsloth_version": None,
        }


@router.get("/api/train/hardware")
def gpu_utilization(_user=Depends(get_current_user)):
    """Live GPU utilization for the training screen."""
    try:
        from utils.hardware.nvidia import get_primary_gpu_utilization
        return get_primary_gpu_utilization()
    except Exception:
        return {"available": False}

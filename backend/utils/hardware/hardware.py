"""
Hardware detection — run once at startup, read everywhere.
Adapted from unsloth-main/studio/backend/utils/hardware/hardware.py
Changes: replaced loggers/structlog with stdlib logging.
"""
import gc
import logging
import os
import platform
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    CUDA = "cuda"
    XPU = "xpu"
    MLX = "mlx"
    CPU = "cpu"


# ── Global state ──
DEVICE: Optional[DeviceType] = None
CHAT_ONLY: bool = True


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _has_torch() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


def _has_mlx() -> bool:
    try:
        import mlx.core
        return True
    except ImportError:
        return False


def detect_hardware() -> DeviceType:
    """Detect best available compute device. Sets DEVICE global. Idempotent."""
    global DEVICE, CHAT_ONLY
    CHAT_ONLY = True

    # 1. CUDA
    if _has_torch():
        import torch
        if torch.cuda.is_available():
            DEVICE = DeviceType.CUDA
            CHAT_ONLY = False
            name = torch.cuda.get_device_properties(0).name
            print(f"Hardware detected: CUDA — {name}")
            return DEVICE

    # 2. XPU (Intel)
    if _has_torch():
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            DEVICE = DeviceType.XPU
            CHAT_ONLY = False
            name = torch.xpu.get_device_name(0)
            print(f"Hardware detected: XPU — {name}")
            return DEVICE

    # 3. MLX (Apple Silicon)
    if is_apple_silicon() and _has_mlx():
        DEVICE = DeviceType.MLX
        chip = platform.processor() or platform.machine()
        print(f"Hardware detected: MLX — Apple Silicon ({chip})")
        return DEVICE

    # 4. CPU fallback
    DEVICE = DeviceType.CPU
    print("Hardware detected: CPU (no GPU backend available)")
    return DEVICE


def get_device() -> Optional[DeviceType]:
    return DEVICE


def get_hardware_state() -> dict:
    """Return current hardware globals as a plain dict (no imports needed)."""
    return {
        "device_type": DEVICE.value if DEVICE is not None else "cpu",
        "chat_only": CHAT_ONLY,
    }


def get_gpu_memory_info() -> Dict[str, Any]:
    """Return GPU memory info dict."""
    if DEVICE == DeviceType.CUDA:
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / 1024 ** 3
            allocated_gb = torch.cuda.memory_allocated(0) / 1024 ** 3
            reserved_gb = torch.cuda.memory_reserved(0) / 1024 ** 3
            free_gb = total_gb - reserved_gb
            util_pct = None
            try:
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                util_pct = u.gpu
            except Exception:
                pass
            return {
                "available": True,
                "backend": "cuda",
                "device": "cuda:0",
                "device_name": props.name,
                "total_gb": round(total_gb, 2),
                "allocated_gb": round(allocated_gb, 2),
                "reserved_gb": round(reserved_gb, 2),
                "free_gb": round(free_gb, 2),
                "utilization_pct": util_pct,
            }
        except Exception as e:
            logger.warning("CUDA memory query failed: %s", e)

    if DEVICE == DeviceType.XPU:
        try:
            import torch
            props = torch.xpu.get_device_properties(0)
            total_gb = props.total_memory / 1024 ** 3
            allocated_gb = torch.xpu.memory_allocated(0) / 1024 ** 3
            return {
                "available": True,
                "backend": "xpu",
                "device": "xpu:0",
                "device_name": props.name,
                "total_gb": round(total_gb, 2),
                "allocated_gb": round(allocated_gb, 2),
                "free_gb": round(total_gb - allocated_gb, 2),
            }
        except Exception:
            pass

    if DEVICE == DeviceType.MLX:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / 1024 ** 3
        return {
            "available": True,
            "backend": "mlx",
            "device": "mps",
            "device_name": f"Apple Silicon ({platform.processor() or platform.machine()})",
            "total_gb": round(total_gb, 2),
            "allocated_gb": 0.0,
            "free_gb": round(mem.available / 1024 ** 3, 2),
        }

    return {"available": False, "backend": "cpu"}


def get_gpu_summary() -> Dict[str, Any]:
    """Compact GPU summary for the hardware endpoint."""
    info = get_gpu_memory_info()
    return {
        "gpu_name": info.get("device_name"),
        "vram_total_gb": info.get("total_gb"),
        "vram_free_gb": info.get("free_gb"),
        "vram_used_gb": info.get("allocated_gb"),
        "device": info.get("backend", "cpu"),
        "available": info.get("available", False),
    }


def get_package_versions() -> Dict[str, Optional[str]]:
    """Get versions of key ML packages."""
    versions: Dict[str, Optional[str]] = {}
    for pkg in ("torch", "transformers", "unsloth", "trl", "peft"):
        try:
            import importlib.metadata as meta
            versions[pkg] = meta.version(pkg)
        except Exception:
            versions[pkg] = None
    # CUDA version
    try:
        import torch
        versions["cuda"] = torch.version.cuda
    except Exception:
        versions["cuda"] = None
    return versions


def clear_gpu_cache() -> None:
    """Clear GPU memory caches."""
    gc.collect()
    if DEVICE == DeviceType.CUDA:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    elif DEVICE == DeviceType.XPU:
        try:
            import torch
            torch.xpu.empty_cache()
        except Exception:
            pass


def apply_gpu_ids(gpu_ids: list | None) -> None:
    """Set CUDA_VISIBLE_DEVICES to the specified GPU IDs in the current process."""
    if not gpu_ids:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)


def prepare_gpu_selection(
    requested_gpu_ids: list | None = None,
    model_name: str = "",
    **kwargs,
) -> tuple[list, dict | None]:
    """
    Simplified GPU selection — returns the first available CUDA GPU.
    Returns (resolved_gpu_ids, gpu_selection_info).
    """
    if not _has_torch():
        return [], None
    try:
        import torch
        if not torch.cuda.is_available():
            return [], None
        count = torch.cuda.device_count()
        if requested_gpu_ids:
            resolved = [g for g in requested_gpu_ids if 0 <= g < count]
        else:
            resolved = [0]  # Default: first GPU
        if not resolved:
            resolved = [0]
        gpu_info = {
            "device": f"cuda:{resolved[0]}",
            "name": torch.cuda.get_device_properties(resolved[0]).name,
            "total_memory_gb": round(
                torch.cuda.get_device_properties(resolved[0]).total_memory / 1024 ** 3, 2
            ),
        }
        return resolved, gpu_info
    except Exception as e:
        logger.warning("GPU selection failed: %s", e)
        return [], None


def safe_num_proc(desired: int) -> int:
    """Safe num_proc — returns 1 on Windows/macOS for spawn-based workers."""
    if platform.system() in ("Windows", "Darwin"):
        return 1
    return min(desired, 4)

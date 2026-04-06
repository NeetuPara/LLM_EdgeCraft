"""
NVIDIA-specific GPU queries via nvidia-smi.
Adapted from unsloth-main — replaced loggers with stdlib logging.
"""
import logging
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _parse_smi_value(raw: str):
    raw = raw.strip()
    if not raw or raw == "[N/A]":
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _build_gpu_metrics(vram_used_mb, vram_total_mb, power_draw, power_limit, **extra) -> dict[str, Any]:
    return {
        **extra,
        "vram_used_gb":         round(vram_used_mb / 1024, 2) if vram_used_mb is not None else None,
        "vram_total_gb":        round(vram_total_mb / 1024, 2) if vram_total_mb is not None else None,
        "vram_utilization_pct": round((vram_used_mb / vram_total_mb) * 100, 1)
                                 if vram_used_mb is not None and vram_total_mb else None,
        "power_draw_w":         power_draw,
        "power_limit_w":        power_limit,
        "power_utilization_pct": round((power_draw / power_limit) * 100, 1)
                                  if power_draw is not None and power_limit else None,
    }


def get_physical_gpu_count() -> Optional[int]:
    """Return physical GPU count via nvidia-smi, or None on failure."""
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return len(result.stdout.strip().splitlines())
    except Exception as e:
        logger.debug("nvidia-smi -L failed: %s", e)
    return None


def get_primary_gpu_utilization() -> dict[str, Any]:
    """Get GPU utilization metrics for the primary GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("nvidia-smi query failed: %s", e)
        return {"available": False}
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False}

    parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 6:
        return {"available": False}

    return _build_gpu_metrics(
        vram_used_mb=_parse_smi_value(parts[2]),
        vram_total_mb=_parse_smi_value(parts[3]),
        power_draw=_parse_smi_value(parts[4]),
        power_limit=_parse_smi_value(parts[5]),
        available=True,
        gpu_utilization_pct=_parse_smi_value(parts[0]),
        temperature_c=_parse_smi_value(parts[1]),
    )

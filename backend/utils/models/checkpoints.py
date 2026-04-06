"""
Checkpoint and LoRA scanning utilities.
Adapted from unsloth-main — replaced loggers with stdlib logging.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _read_checkpoint_loss(checkpoint_path: Path) -> Optional[float]:
    trainer_state = checkpoint_path / "trainer_state.json"
    if not trainer_state.exists():
        return None
    try:
        with open(trainer_state) as f:
            state = json.load(f)
        log_history = state.get("log_history", [])
        if log_history:
            return log_history[-1].get("loss")
    except Exception as e:
        logger.debug("Could not read loss from %s: %s", trainer_state, e)
    return None


def scan_checkpoints(
    outputs_dir: Optional[str] = None,
) -> List[Tuple[str, List[Tuple[str, str, Optional[float]]], dict]]:
    """
    Scan outputs folder for training runs and their checkpoints.

    Returns: [(model_name, [(display_name, path, loss), ...], metadata), ...]
    """
    if outputs_dir is None:
        from utils.paths import outputs_root
        outputs_dir = str(outputs_root())

    from utils.paths import resolve_output_dir
    outputs_path = resolve_output_dir(outputs_dir)

    if not outputs_path.exists():
        return []

    models = []
    try:
        for item in outputs_path.iterdir():
            if not item.is_dir():
                continue
            config_file = item / "config.json"
            adapter_config = item / "adapter_config.json"
            if not (config_file.exists() or adapter_config.exists()):
                continue

            metadata: dict = {}
            try:
                if adapter_config.exists():
                    cfg = json.loads(adapter_config.read_text())
                    metadata["base_model"] = cfg.get("base_model_name_or_path")
                    metadata["peft_type"] = cfg.get("peft_type")
                    metadata["lora_rank"] = cfg.get("r")
                elif config_file.exists():
                    cfg = json.loads(config_file.read_text())
                    metadata["base_model"] = cfg.get("_name_or_path")
            except Exception:
                pass

            # Fallback: extract base model from folder name
            if not metadata.get("base_model"):
                parts = item.name.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    name_part = parts[0]
                    idx = name_part.find("_")
                    if idx > 0:
                        metadata["base_model"] = name_part[:idx] + "/" + name_part[idx + 1:]
                    else:
                        metadata["base_model"] = name_part

            checkpoints: List[Tuple[str, str, Optional[float]]] = []
            checkpoints.append((item.name, str(item), None))

            for sub in sorted(item.iterdir()):
                if not sub.is_dir() or not sub.name.startswith("checkpoint-"):
                    continue
                if (sub / "config.json").exists() or (sub / "adapter_config.json").exists():
                    loss = _read_checkpoint_loss(sub)
                    checkpoints.append((sub.name, str(sub), loss))

            # Assign last checkpoint's loss to the main entry
            if len(checkpoints) > 1:
                last_loss = checkpoints[-1][2]
                checkpoints[0] = (checkpoints[0][0], checkpoints[0][1], last_loss)

            models.append((item.name, checkpoints, metadata))

        # Sort newest first
        models.sort(key=lambda x: Path(x[1][0][1]).stat().st_mtime, reverse=True)
        return models
    except Exception as e:
        logger.error("Error scanning checkpoints: %s", e)
        return []


def scan_trained_loras(outputs_dir: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Scan outputs folder for LoRA adapters.
    Returns: [(display_name, adapter_path), ...]
    """
    if outputs_dir is None:
        from utils.paths import outputs_root
        outputs_dir = str(outputs_root())

    from utils.paths import resolve_output_dir
    outputs_path = resolve_output_dir(outputs_dir)

    if not outputs_path.exists():
        return []

    loras = []
    try:
        for item in outputs_path.iterdir():
            if not item.is_dir():
                continue
            if (item / "adapter_config.json").exists() or (item / "adapter_model.safetensors").exists():
                loras.append((item.name, str(item)))
        loras.sort(key=lambda x: Path(x[1]).stat().st_mtime, reverse=True)
        return loras
    except Exception as e:
        logger.error("Error scanning LoRAs: %s", e)
        return []

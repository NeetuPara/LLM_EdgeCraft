"""
Config file parser — parses .yaml / .cfg / .txt training config files
and returns a flat dict with frontend-compatible camelCase keys.
"""
import io
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from auth.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/config", tags=["config"])

_ALLOWED_EXTS = {".yaml", ".yml", ".cfg", ".ini", ".txt"}
_ALLOWED_EXTS_INFERENCE = {".yaml", ".yml", ".cfg", ".ini", ".txt", ".json"}

# ── Inference key map: config-file key → frontend camelCase field ──
_INFERENCE_KEY_MAP: dict[str, str] = {
    "temperature":          "temperature",
    "temp":                 "temperature",
    "top_p":                "topP",
    "topp":                 "topP",
    "top_k":                "topK",
    "topk":                 "topK",
    "min_p":                "minP",
    "minp":                 "minP",
    "max_tokens":           "maxTokens",
    "max_new_tokens":       "maxTokens",
    "max_length":           "maxTokens",
    "repetition_penalty":   "repetitionPenalty",
    "rep_penalty":          "repetitionPenalty",
    "repeat_penalty":       "repetitionPenalty",
    "system_prompt":        "systemPrompt",
    "system":               "systemPrompt",
    # greedy=true → temperature=0; greedy=false → leave temperature as-is
    # (handled in the endpoint below via special-case logic)
}

# ── Key normalisation map: config-file key → frontend camelCase field ──
_KEY_MAP: dict[str, str] = {
    # epochs / steps
    "num_epochs":                   "numEpochs",
    "epochs":                       "numEpochs",
    "max_steps":                    "maxSteps",
    # learning rate
    "learning_rate":                "learningRate",
    "lr":                           "learningRate",
    # batch
    "batch_size":                   "batchSize",
    "per_device_train_batch_size":  "batchSize",
    # context
    "max_seq_length":               "maxSeqLength",
    "max_sequence_length":          "maxSeqLength",
    "context_length":               "maxSeqLength",
    # scheduler
    "lr_scheduler_type":            "lrScheduler",
    "lr_scheduler":                 "lrScheduler",
    # LoRA
    "lora_r":                       "loraR",
    "r":                            "loraR",
    "lora_alpha":                   "loraAlpha",
    "alpha":                        "loraAlpha",
    "lora_dropout":                 "loraDropout",
    "use_rslora":                   "useRslora",
    "rslora":                       "useRslora",
    "target_modules":               "targetModules",
    "use_loftq":                    "useLoftq",
    # advanced
    "gradient_accumulation_steps":  "gradAccumSteps",
    "grad_accum_steps":             "gradAccumSteps",
    "grad_accumulation":            "gradAccumSteps",
    "warmup_steps":                 "warmupSteps",
    "weight_decay":                 "weightDecay",
    "optimizer":                    "optimizer",
    "optim":                        "optimizer",
    "packing":                      "packing",
    "train_on_completions":         "trainOnCompletions",
    "save_steps":                   "saveSteps",
    "save_strategy":                "saveStrategy",
    "eval_steps":                   "evalSteps",
    "gradient_checkpointing":       "gradientCheckpointing",
    # logging
    "enable_wandb":                 "enableWandB",
    "wandb_project":                "wandbProject",
    "enable_tensorboard":           "enableTensorBoard",
}


def _coerce(value):
    """Convert string values to appropriate Python types."""
    if not isinstance(value, str):
        return value
    low = value.strip().lower()
    if low in ("true", "yes", "1"):
        return True
    if low in ("false", "no", "0"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip()


def _parse_yaml(content: str) -> dict:
    try:
        import yaml
        data = yaml.safe_load(content)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        raise HTTPException(400, f"YAML parse error: {e}")


def _parse_cfg(content: str) -> dict:
    """configparser — supports [section] headers; merges all sections."""
    import configparser
    cp = configparser.ConfigParser()
    try:
        cp.read_string(content)
    except Exception as e:
        raise HTTPException(400, f"CFG parse error: {e}")
    result: dict = {}
    # Include DEFAULT section values
    result.update({k: _coerce(v) for k, v in cp.defaults().items()})
    for section in cp.sections():
        for key, val in cp.items(section):
            result[key] = _coerce(val)
    return result


def _parse_txt(content: str) -> dict:
    """Simple key=value or key:value per line, ignoring # comments."""
    result: dict = {}
    for line in content.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        for sep in ("=", ":"):
            if sep in line:
                key, _, val = line.partition(sep)
                result[key.strip().lower()] = _coerce(val.strip())
                break
    return result


def _parse_json(content: str) -> dict:
    import json
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        raise HTTPException(400, f"JSON parse error: {e}")


def _normalise(raw: dict) -> dict:
    """Map config keys → frontend camelCase field names, drop unknown keys."""
    out: dict = {}
    for raw_key, value in raw.items():
        normalised_key = raw_key.strip().lower().replace("-", "_").replace(" ", "_")
        frontend_key = _KEY_MAP.get(normalised_key)
        if frontend_key:
            out[frontend_key] = value
    return out


@router.post("/parse")
async def parse_config(file: UploadFile, _user=Depends(get_current_user)):
    """
    Parse a training config file and return frontend-compatible field values.

    Supported formats:
      .yaml / .yml  — YAML key: value pairs
      .cfg / .ini   — INI-style [section] key = value
      .txt          — plain key = value or key: value lines
    """
    filename = (file.filename or "config").lower()
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTS))}"
        )

    content = (await file.read()).decode("utf-8", errors="replace")
    if not content.strip():
        raise HTTPException(400, "Config file is empty")

    if ext in (".yaml", ".yml"):
        raw = _parse_yaml(content)
    elif ext in (".cfg", ".ini"):
        raw = _parse_cfg(content)
    else:
        raw = _parse_txt(content)

    config = _normalise(raw)
    if not config:
        raise HTTPException(422, "No recognised config keys found in file. Check the template for valid key names.")

    logger.info("Config parsed from %s: %d keys applied", filename, len(config))
    return {"config": config, "applied_count": len(config), "filename": file.filename}


@router.post("/parse-inference")
async def parse_inference_config(file: UploadFile, _user=Depends(get_current_user)):
    """
    Parse an inference config file and return frontend-compatible InferenceParams values.

    Supported formats:
      .json           — JSON object
      .yaml / .yml    — YAML key: value pairs
      .cfg / .ini     — INI-style key = value
      .txt            — plain key = value or key: value lines

    Recognised keys: temperature, top_p, top_k, min_p, max_tokens,
                     repetition_penalty, system_prompt (and aliases)
    """
    filename = (file.filename or "config").lower()
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTS_INFERENCE:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTS_INFERENCE))}"
        )

    content = (await file.read()).decode("utf-8", errors="replace")
    if not content.strip():
        raise HTTPException(400, "Config file is empty")

    if ext == ".json":
        raw = _parse_json(content)
    elif ext in (".yaml", ".yml"):
        raw = _parse_yaml(content)
    elif ext in (".cfg", ".ini"):
        raw = _parse_cfg(content)
    else:
        raw = _parse_txt(content)

    # Normalise to inference camelCase keys
    out: dict = {}
    for raw_key, value in raw.items():
        normalised = raw_key.strip().lower().replace("-", "_").replace(" ", "_")
        frontend_key = _INFERENCE_KEY_MAP.get(normalised)
        if frontend_key:
            coerced = _coerce(value) if isinstance(value, str) else value
            # "greedy = true" means do_sample = false (inverted)
            if normalised == "greedy":
                coerced = not bool(coerced)
            out[frontend_key] = coerced

    if not out:
        raise HTTPException(422, "No recognised inference keys found. Valid keys: temperature, top_p, top_k, min_p, max_tokens, repetition_penalty, system_prompt")

    logger.info("Inference config parsed from %s: %d keys applied", filename, len(out))
    return {"config": out, "applied_count": len(out), "filename": file.filename}

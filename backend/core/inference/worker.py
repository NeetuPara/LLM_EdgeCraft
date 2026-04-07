"""
Inference subprocess entry point.

Spawned once per model load. Stays alive accepting generate commands
until shutdown. Pattern mirrors core/training/worker.py.

Commands (cmd_queue):
    {"type": "generate", "request_id": str, "messages": [...], ...gen_params, "use_adapter": bool|None}
    {"type": "shutdown"}

Responses (resp_queue):
    {"type": "loaded", "success": True, "model_info": {...}}
    {"type": "token", "text": str, "request_id": str}
    {"type": "gen_done", "request_id": str}
    {"type": "gen_error", "error": str, "request_id": str}
    {"type": "error", "error": str}
    {"type": "status", "message": str}
"""

from __future__ import annotations

import logging
import os
import queue as _queue
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _setup_paths() -> None:
    """Add backend root and compiled cache to sys.path."""
    backend_dir = str(Path(__file__).resolve().parent.parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    # Unsloth compiled cache — must be on PYTHONPATH before any imports
    _compile_cache = os.environ.get("UNSLOTH_COMPILE_LOCATION", "")
    if not _compile_cache:
        _compile_cache = os.path.join(backend_dir, "unsloth_compiled_cache")
    elif not os.path.isabs(_compile_cache):
        _compile_cache = os.path.abspath(_compile_cache)
    os.environ["UNSLOTH_COMPILE_LOCATION"] = _compile_cache
    _pp = os.environ.get("PYTHONPATH", "")
    if _compile_cache not in _pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = _compile_cache + (os.pathsep + _pp if _pp else "")
    if _compile_cache not in sys.path:
        sys.path.insert(0, _compile_cache)


def _send(resp_queue: Any, msg: dict) -> None:
    try:
        resp_queue.put(msg)
    except Exception:
        pass


def run_inference_process(
    *,
    cmd_queue: Any,
    resp_queue: Any,
    cancel_event: Any,
    config: dict,
) -> None:
    """Subprocess entry point — loads model then enters command loop."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = "ignore"

    import warnings
    warnings.filterwarnings("ignore")

    _setup_paths()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    # Apply GPU selection
    from utils.hardware import apply_gpu_ids
    apply_gpu_ids(config.get("resolved_gpu_ids"))

    # ── Load model ──
    model_path = config["model_path"]
    model_short = model_path.split("/")[-1].split("\\")[-1]
    is_adapter = (Path(model_path) / "adapter_config.json").exists()

    _send(resp_queue, {"type": "status", "message": f"Starting {model_short}..."})

    try:
        from core.inference.inference_backend import InferenceBackend

        _send(resp_queue, {"type": "status", "message": "Importing Unsloth..."})
        backend = InferenceBackend()

        _send(resp_queue, {"type": "status",
            "message": f"{'Loading adapter + base model' if is_adapter else 'Loading model weights'} ({model_short})..."})
        backend.load_model(
            model_path=config["model_path"],
            load_in_4bit=config.get("load_in_4bit", True),
            max_seq_length=config.get("max_seq_length", 2048),
            hf_token=config.get("hf_token") or None,
        )

        # ── Kernel warmup ──────────────────────────────────────────────────
        # Triton / CUDA kernels JIT-compile on their FIRST use.
        # Without warmup: user's first query → 20-60s delay while kernels compile.
        # With warmup:    kernels compile NOW (hidden inside loading), first query instant.
        # Compiled kernels are cached in unsloth_compiled_cache/ — warmup is fast on
        # subsequent server restarts (cache hit), slow only on the very first ever run.
        # ──────────────────────────────────────────────────────────────────
        _send(resp_queue, {"type": "status", "message": "Warming up kernels..."})
        backend.warmup_generation()

        _send(resp_queue, {
            "type": "loaded",
            "success": True,
            "model_info": {
                "model_path":             config["model_path"],
                "is_lora":                backend.is_lora,
                "training_system_prompt": backend.training_system_prompt,
                "training_inference_params": backend.training_inference_params,
            },
        })
        logger.info("Model loaded, entering command loop")

    except Exception as exc:
        _send(resp_queue, {
            "type": "loaded",
            "success": False,
            "error": str(exc),
        })
        logger.error("Failed to load model: %s", exc, exc_info=True)
        return

    # ── Command loop ──
    while True:
        try:
            cmd = cmd_queue.get(timeout=1.0)
        except _queue.Empty:
            continue
        except (EOFError, OSError):
            break

        ctype = cmd.get("type", "")

        if ctype == "shutdown":
            logger.info("Shutdown command received")
            break

        elif ctype == "generate":
            request_id = cmd.get("request_id", "")
            messages = cmd.get("messages", [])
            use_adapter = cmd.get("use_adapter")  # None | True | False
            cancel_event.clear()

            try:
                token_count = 0
                for token in backend.generate_stream(
                    messages=messages,
                    system_prompt=cmd.get("system_prompt", ""),
                    temperature=cmd.get("temperature", 0.7),
                    top_p=cmd.get("top_p", 0.9),
                    top_k=cmd.get("top_k", 40),
                    min_p=cmd.get("min_p", 0.0),
                    max_new_tokens=cmd.get("max_new_tokens", 512),
                    repetition_penalty=cmd.get("repetition_penalty", 1.1),
                    use_adapter=use_adapter,
                    cancel_event=cancel_event,
                ):
                    token_count += 1
                    _send(resp_queue, {"type": "token", "text": token, "request_id": request_id})
                    if cancel_event.is_set():
                        break

                logger.info("Worker: generation done, sent %d tokens", token_count)
                _send(resp_queue, {"type": "gen_done", "request_id": request_id})

            except Exception as exc:
                logger.error("Generation error: %s", exc, exc_info=True)
                _send(resp_queue, {
                    "type": "gen_error",
                    "error": str(exc),
                    "request_id": request_id,
                })

        else:
            logger.warning("Unknown command type: %s", ctype)

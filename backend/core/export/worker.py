"""
Export subprocess entry point — self-contained.
Uses the installed `unsloth` package directly.
No imports from unsloth-main's studio backend.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _setup_paths() -> None:
    our_backend = str(Path(__file__).resolve().parent.parent.parent)
    if our_backend not in sys.path:
        sys.path.insert(0, our_backend)


def _send_response(resp_queue: Any, response: dict) -> None:
    try:
        resp_queue.put(response)
    except (OSError, ValueError) as exc:
        logger.error("Failed to send response: %s", exc)


class _ExportBackend:
    """Self-contained export backend using installed unsloth package."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_vision = False
        self.is_peft = False

    def load_checkpoint(self, checkpoint_path: str, max_seq_length: int = 2048,
                        load_in_4bit: bool = True, trust_remote_code: bool = False):
        from unsloth import FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
        )
        self.is_peft = hasattr(self.model, "peft_config") or (
            Path(checkpoint_path) / "adapter_config.json").exists()
        return True, f"Loaded: {checkpoint_path}"

    def export_merged_model(self, save_directory, format_type="16-bit (FP16)",
                            push_to_hub=False, repo_id=None, hf_token=None, private=False):
        method = "merged_16bit" if "16" in format_type else "merged_4bit"
        self.model.save_pretrained_merged(save_directory, self.tokenizer, save_method=method,
                                          push_to_hub=push_to_hub, token=hf_token)
        return True, f"Merged model saved to {save_directory}"

    def export_base_model(self, save_directory, push_to_hub=False, repo_id=None,
                          hf_token=None, private=False, base_model_id=None):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        return True, f"Base model saved to {save_directory}"

    def export_gguf(self, save_directory, quantization_method="Q4_K_M",
                    push_to_hub=False, repo_id=None, hf_token=None):
        quant = quantization_method.lower().replace("-", "_")
        self.model.save_pretrained_gguf(save_directory, self.tokenizer,
                                        quantization_method=quant,
                                        push_to_hub=push_to_hub, token=hf_token)
        return True, f"GGUF ({quantization_method}) saved to {save_directory}"

    def export_lora_adapter(self, save_directory, push_to_hub=False, repo_id=None,
                            hf_token=None, private=False):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        if push_to_hub and hf_token:
            self.model.push_to_hub(repo_id or save_directory, token=hf_token, private=private)
            self.tokenizer.push_to_hub(repo_id or save_directory, token=hf_token, private=private)
        return True, f"LoRA adapter saved to {save_directory}"

    def cleanup_memory(self):
        del self.model; del self.tokenizer
        self.model = None; self.tokenizer = None
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
        return True


def run_export_process(*, cmd_queue: Any, resp_queue: Any, config: dict) -> None:
    """Subprocess entry point."""
    import queue as _queue

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = "ignore"
    import warnings; warnings.filterwarnings("ignore")

    _setup_paths()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if sys.platform == "win32":
        try:
            import triton  # noqa
        except ImportError:
            os.environ["TORCHDYNAMO_DISABLE"] = "1"

    try:
        _send_response(resp_queue, {"type": "status", "message": "Importing Unsloth...", "ts": time.time()})
        import unsloth  # noqa
    except Exception as exc:
        _send_response(resp_queue, {"type": "error", "error": f"Import failed: {exc}",
                                    "stack": traceback.format_exc(limit=20), "ts": time.time()})
        return

    backend = _ExportBackend()
    try:
        ok, msg = backend.load_checkpoint(
            config["checkpoint_path"], config.get("max_seq_length", 2048),
            config.get("load_in_4bit", True), config.get("trust_remote_code", False))
        _send_response(resp_queue, {"type": "loaded", "success": ok, "message": msg,
                                    "checkpoint": config["checkpoint_path"] if ok else None,
                                    "is_vision": backend.is_vision, "is_peft": backend.is_peft, "ts": time.time()})
    except Exception as exc:
        _send_response(resp_queue, {"type": "loaded", "success": False, "message": str(exc), "ts": time.time()})
        return

    while True:
        try:
            cmd = cmd_queue.get(timeout=1.0)
        except _queue.Empty:
            continue
        except (EOFError, OSError):
            return
        if cmd is None:
            continue
        cmd_type = cmd.get("type", "")
        try:
            if cmd_type == "export":
                et = cmd["export_type"]
                rt = f"export_{et}_done"
                if et == "merged":
                    ok, msg = backend.export_merged_model(cmd.get("save_directory", ""), cmd.get("format_type", "16-bit (FP16)"),
                                                          cmd.get("push_to_hub", False), cmd.get("repo_id"), cmd.get("hf_token"))
                elif et == "base":
                    ok, msg = backend.export_base_model(cmd.get("save_directory", ""), cmd.get("push_to_hub", False),
                                                        cmd.get("repo_id"), cmd.get("hf_token"))
                elif et == "gguf":
                    ok, msg = backend.export_gguf(cmd.get("save_directory", ""), cmd.get("quantization_method", "Q4_K_M"),
                                                   cmd.get("push_to_hub", False), cmd.get("repo_id"), cmd.get("hf_token"))
                elif et == "lora":
                    ok, msg = backend.export_lora_adapter(cmd.get("save_directory", ""), cmd.get("push_to_hub", False),
                                                          cmd.get("repo_id"), cmd.get("hf_token"))
                else:
                    ok, msg = False, f"Unknown: {et}"
                _send_response(resp_queue, {"type": rt, "success": ok, "message": msg, "ts": time.time()})
            elif cmd_type == "cleanup":
                _send_response(resp_queue, {"type": "cleanup_done", "success": backend.cleanup_memory(), "ts": time.time()})
            elif cmd_type == "shutdown":
                backend.cleanup_memory(); return
        except Exception as exc:
            _send_response(resp_queue, {"type": "error", "error": str(exc),
                                        "stack": traceback.format_exc(limit=20), "ts": time.time()})

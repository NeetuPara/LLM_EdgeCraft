"""
Inference orchestrator — subprocess-based (mirrors unsloth-main approach).

Two backends:
  HF / LoRA  → persistent subprocess (worker.py + inference_backend.py)
               FastLanguageModel.from_pretrained, TextIteratorStreamer
  GGUF       → llama-server OS subprocess (same as before)

The HF path supports compare mode: pass use_adapter=False for base model
output, use_adapter=True to re-enable LoRA layers — same weights in VRAM.
"""

from __future__ import annotations

import json as _json
import logging
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Generator, Optional

import httpx

logger = logging.getLogger(__name__)

_CTX = mp.get_context("spawn")

_manager: Optional["InferenceManager"] = None
_manager_lock = threading.Lock()


def get_inference_manager() -> "InferenceManager":
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = InferenceManager()
    return _manager


_compare_manager: Optional["InferenceManager"] = None
_compare_manager_lock = threading.Lock()


def get_compare_manager() -> "InferenceManager":
    """Second inference manager for compare-mode right slot."""
    global _compare_manager
    if _compare_manager is None:
        with _compare_manager_lock:
            if _compare_manager is None:
                _compare_manager = InferenceManager()
    return _compare_manager


# ══════════════════════════════════════════════════════
# llama-server helpers (GGUF path)
# ══════════════════════════════════════════════════════

def _find_llama_server() -> Optional[str]:
    import shutil
    for env_key in ("LLAMA_SERVER_PATH", "UNSLOTH_LLAMA_CPP_PATH"):
        env = os.environ.get(env_key)
        if env:
            p = Path(env)
            if p.is_file():
                return str(p)
            for sub in ["llama-server", "llama-server.exe",
                        "build/bin/llama-server", "build/bin/Release/llama-server.exe"]:
                c = p / sub
                if c.exists():
                    return str(c)
    for base in [Path.home() / ".unsloth" / "llama.cpp",
                 Path.home() / ".unslothcraft" / "llama.cpp"]:
        for sub in ["llama-server", "llama-server.exe",
                    "build/bin/llama-server", "build/bin/Release/llama-server.exe"]:
            c = base / sub
            if c.exists():
                return str(c)
    return shutil.which("llama-server")


# ══════════════════════════════════════════════════════
# InferenceManager
# ══════════════════════════════════════════════════════

class InferenceManager:
    """
    Unified inference backend.
    Detects GGUF vs HF/LoRA and routes to the correct backend.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # ── HF subprocess state ──
        self._hf_proc: Optional[mp.Process] = None
        self._hf_cmd_queue: Any = None
        self._hf_resp_queue: Any = None
        self._hf_cancel_event: Any = None
        self._hf_gen_lock = threading.Lock()  # Serialize generation calls

        # ── GGUF llama-server state ──
        self._llama_proc: Optional[subprocess.Popen] = None
        self._llama_port: int = 8899

        # ── Common state ──
        self._active_model: Optional[str] = None
        self._is_gguf: bool = False
        self._is_lora: bool = False
        self._context_length: Optional[int] = None
        self._loading: bool = False
        self._loading_model: Optional[str] = None

    # ── Properties ──

    @property
    def is_loaded(self) -> bool:
        return self._active_model is not None

    @property
    def active_model(self) -> Optional[str]:
        return self._active_model

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def loading_model(self) -> Optional[str]:
        return self._loading_model

    def get_status(self) -> dict:
        return {
            "status": "loaded" if self.is_loaded else ("loading" if self._loading else "idle"),
            "model_name": self._active_model,
            "is_gguf": self._is_gguf,
            "is_lora": self._is_lora,
            "context_length": self._context_length,
            "loading_model": self._loading_model if self._loading else None,
        }

    # ── GGUF detection ──

    def _detect_gguf(self, model_path: str, gguf_variant: Optional[str]) -> bool:
        if gguf_variant:
            return True
        lower = model_path.lower()
        return lower.endswith(".gguf") or "gguf" in lower

    # ── Unload helpers ──

    def _shutdown_hf_subprocess(self, timeout: float = 10.0) -> None:
        if self._hf_proc is None:
            return
        if self._hf_cancel_event is not None:
            self._hf_cancel_event.set()
        try:
            if self._hf_cmd_queue is not None:
                self._hf_cmd_queue.put({"type": "shutdown"})
        except Exception:
            pass
        try:
            self._hf_proc.join(timeout=timeout)
        except Exception:
            pass
        if self._hf_proc.is_alive():
            self._hf_proc.terminate()
            self._hf_proc.join(timeout=5)
        self._hf_proc = None
        self._hf_cmd_queue = None
        self._hf_resp_queue = None
        self._hf_cancel_event = None

    def _kill_llama_server(self) -> None:
        if self._llama_proc is not None:
            try:
                self._llama_proc.terminate()
                self._llama_proc.wait(timeout=10)
            except Exception:
                try:
                    self._llama_proc.kill()
                except Exception:
                    pass
            self._llama_proc = None

    def unload(self) -> dict:
        self._shutdown_hf_subprocess()
        self._kill_llama_server()
        with self._lock:
            self._active_model = None
            self._is_gguf = False
            self._is_lora = False
        return {"status": "unloaded"}

    # ── Load ──

    async def load(
        self,
        model_path: str,
        gguf_variant: Optional[str] = None,
        hf_token: Optional[str] = None,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        **kwargs,
    ) -> dict:
        import asyncio

        with self._lock:
            if self._loading:
                return {"status": "already_loading", "model_name": self._loading_model}
            self._loading = True
            self._loading_model = model_path

        try:
            is_gguf = self._detect_gguf(model_path, gguf_variant)

            # Unload any existing model first
            self._shutdown_hf_subprocess(timeout=5)
            self._kill_llama_server()

            if is_gguf:
                result = await asyncio.to_thread(
                    self._load_gguf_sync, model_path, gguf_variant, hf_token, max_seq_length
                )
            else:
                result = await asyncio.to_thread(
                    self._load_hf_sync, model_path, hf_token, max_seq_length, load_in_4bit
                )

            with self._lock:
                if result.get("status") == "loaded":
                    self._active_model = model_path
                    self._is_gguf = is_gguf
                    self._is_lora = result.get("is_lora", False)
                    self._context_length = result.get("context_length")

            return result

        finally:
            with self._lock:
                self._loading = False
                self._loading_model = None

    # ── HF subprocess load ──

    def _load_hf_sync(
        self,
        model_path: str,
        hf_token: Optional[str],
        max_seq_length: int,
        load_in_4bit: bool,
    ) -> dict:
        from core.inference.worker import run_inference_process
        from utils.hardware import apply_gpu_ids

        config = {
            "model_path": model_path,
            "load_in_4bit": load_in_4bit,
            "max_seq_length": max_seq_length,
            "hf_token": hf_token or "",
            "resolved_gpu_ids": None,
        }

        self._hf_cmd_queue = _CTX.Queue()
        self._hf_resp_queue = _CTX.Queue()
        self._hf_cancel_event = _CTX.Event()

        proc = _CTX.Process(
            target=run_inference_process,
            kwargs={
                "cmd_queue": self._hf_cmd_queue,
                "resp_queue": self._hf_resp_queue,
                "cancel_event": self._hf_cancel_event,
                "config": config,
            },
            daemon=True,
        )
        proc.start()
        self._hf_proc = proc
        logger.info("Inference subprocess started (pid=%s) for %s", proc.pid, model_path)

        # Wait for "loaded" response — up to 10 minutes (large models take time)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if not proc.is_alive():
                return {"status": "error", "error": "Inference subprocess exited unexpectedly during load"}
            try:
                resp = self._hf_resp_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            rtype = resp.get("type", "")
            if rtype == "status":
                logger.info("Inference subprocess: %s", resp.get("message", ""))
                continue
            if rtype == "loaded":
                if resp.get("success"):
                    info = resp.get("model_info", {})
                    return {
                        "status": "loaded",
                        "model_name": model_path,
                        "is_lora": info.get("is_lora", False),
                        "is_gguf": False,
                        "context_length": None,
                        # Training metadata — pre-populate chat Settings
                        "training_system_prompt":    info.get("training_system_prompt", ""),
                        "training_inference_params": info.get("training_inference_params", {}),
                    }
                else:
                    return {"status": "error", "error": resp.get("error", "Load failed")}

        self._shutdown_hf_subprocess(timeout=5)
        return {"status": "error", "error": "Timeout waiting for model to load (600s)"}

    # ── GGUF llama-server load ──

    def _load_gguf_sync(
        self,
        model_path: str,
        gguf_variant: Optional[str],
        hf_token: Optional[str],
        max_seq_length: int,
    ) -> dict:
        binary = _find_llama_server()
        if not binary:
            return {
                "status": "error",
                "error": "llama-server not found. Install llama.cpp or set LLAMA_SERVER_PATH.",
            }

        self._kill_llama_server()

        # Resolve GGUF file
        gguf_path = None
        local = Path(model_path)
        if local.exists() and local.suffix.lower() == ".gguf":
            gguf_path = str(local)
        else:
            try:
                from huggingface_hub import hf_hub_download
                variant = gguf_variant or self._guess_gguf_variant(model_path)
                if not variant:
                    return {"status": "error", "error": f"No GGUF variant found for {model_path}"}
                gguf_path = hf_hub_download(repo_id=model_path, filename=variant, token=hf_token)
                logger.info("Downloaded GGUF %s → %s", variant, gguf_path)
            except Exception as e:
                return {"status": "error", "error": f"GGUF download failed: {e}"}

        if not gguf_path or not Path(gguf_path).exists():
            return {"status": "error", "error": f"GGUF file not found: {gguf_path}"}

        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            self._llama_port = s.getsockname()[1]

        cmd = [binary, "-m", gguf_path, "--port", str(self._llama_port),
               "-c", str(max_seq_length), "--parallel", "1", "--flash-attn"]

        logger.info("Starting llama-server on port %d", self._llama_port)
        kw: dict = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            self._llama_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw
            )
        except Exception as e:
            return {"status": "error", "error": f"Failed to start llama-server: {e}"}

        url = f"http://127.0.0.1:{self._llama_port}/health"
        for _ in range(120):
            time.sleep(0.5)
            if self._llama_proc.poll() is not None:
                out = (self._llama_proc.stdout.read() or b"").decode(errors="replace")[-500:]
                return {"status": "error", "error": f"llama-server exited: {out}"}
            try:
                if httpx.get(url, timeout=2).status_code == 200:
                    logger.info("llama-server ready on port %d", self._llama_port)
                    return {
                        "status": "loaded",
                        "model_name": model_path,
                        "is_gguf": True,
                        "is_lora": False,
                        "context_length": max_seq_length,
                    }
            except Exception:
                pass

        self._kill_llama_server()
        return {"status": "error", "error": "llama-server did not start within 60s"}

    def _guess_gguf_variant(self, repo_id: str) -> Optional[str]:
        try:
            from huggingface_hub import HfApi
            files = list(HfApi().list_repo_files(repo_id))
            gguf = [f for f in files if f.endswith(".gguf")]
            for f in gguf:
                if "Q4_K_M" in f or "q4_k_m" in f:
                    return f
            return gguf[0] if gguf else None
        except Exception:
            return None

    # ── stream_completions ──

    def stream_completions(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.05,
        max_tokens: int = 512,
        repetition_penalty: float = 1.1,
        presence_penalty: float = 0.0,
        system_prompt: Optional[str] = None,
        cancel_event=None,
        use_adapter: Optional[bool] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        if not self.is_loaded:
            raise RuntimeError("No model loaded.")

        if self._is_gguf:
            # llama.cpp: temperature=0 triggers greedy — same convention as HF path
            yield from self._stream_from_llama_server(
                messages, temperature, top_p, top_k, min_p,
                max_tokens, repetition_penalty, presence_penalty,
                system_prompt, cancel_event,
            )
        else:
            yield from self._stream_from_hf_subprocess(
                messages=messages,
                system_prompt=system_prompt or "",
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_new_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                use_adapter=use_adapter,
                cancel_event=cancel_event,
            )

    # ── HF subprocess generation ──

    def _stream_from_hf_subprocess(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.1,
        use_adapter: Optional[bool] = None,
        cancel_event=None,
    ) -> Generator[str, None, None]:
        if self._hf_proc is None or not self._hf_proc.is_alive():
            raise RuntimeError("Inference subprocess is not running")

        import uuid
        request_id = str(uuid.uuid4())

        cmd = {
            "type": "generate",
            "request_id": request_id,
            "messages": messages,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "use_adapter": use_adapter,
        }

        with self._hf_gen_lock:
            if self._hf_cancel_event is not None:
                self._hf_cancel_event.clear()

            try:
                self._hf_cmd_queue.put(cmd)
            except Exception as e:
                raise RuntimeError(f"Failed to send generate command: {e}")

            # Read tokens until gen_done or gen_error
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    if self._hf_cancel_event is not None:
                        self._hf_cancel_event.set()
                    break

                if not self._hf_proc.is_alive():
                    raise RuntimeError("Inference subprocess crashed during generation")

                try:
                    resp = self._hf_resp_queue.get(timeout=30.0)
                except queue.Empty:
                    continue

                rtype = resp.get("type", "")

                if rtype == "token":
                    token = resp.get("text", "")
                    if token:
                        yield token

                elif rtype == "gen_done":
                    break

                elif rtype == "gen_error":
                    raise RuntimeError(resp.get("error", "Generation error"))

                elif rtype == "status":
                    continue

    # ── GGUF llama-server generation ──

    def _stream_from_llama_server(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        max_tokens: int,
        rep_pen: float,
        pres_pen: float,
        system_prompt: Optional[str],
        cancel_event,
    ) -> Generator[str, None, None]:
        full_msgs = []
        if system_prompt:
            full_msgs.append({"role": "system", "content": system_prompt})
        full_msgs.extend(messages)

        url = f"http://127.0.0.1:{self._llama_port}/v1/chat/completions"
        body = {
            "messages": full_msgs,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "repeat_penalty": rep_pen,
            "presence_penalty": pres_pen,
        }
        try:
            with httpx.stream("POST", url, json=body, timeout=120.0) as resp:
                for line in resp.iter_lines():
                    if cancel_event and cancel_event.is_set():
                        break
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = _json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except _json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error("llama-server stream error: %s", e)
            raise RuntimeError(f"Inference error: {e}")

"""
InferenceBackend — runs inside the inference subprocess.
Uses FastLanguageModel for HF/LoRA models (same pattern as unsloth-main).
Never imported in the main process.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)


class InferenceBackend:
    """
    Handles model loading and text generation inside the inference subprocess.
    Mirrors unsloth-main's InferenceBackend but stripped to text-only needs.

    LoRA adapters: FastLanguageModel.from_pretrained(adapter_path) auto-detects
    adapter_config.json and loads base model + merges adapters.

    Compare mode: disable_adapter_layers() / enable_adapter_layers() on the
    loaded PeftModel to get base vs LoRA output from the same weights.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path: Optional[str] = None
        self.is_lora: bool = False
        self.max_seq_length: int = 2048
        self._generation_lock = threading.Lock()
        # System prompt saved during training — auto-applied for LoRA models
        self.training_system_prompt: str = ""

    # ── Load ──

    def _detect_vlm_path(self, model_path: str) -> bool:
        """
        Heuristic: is this model path a VLM?
        Checks model name keywords AND the base_model_name in adapter_config.json
        (so checkpoints of SmolVLM fine-tunes are also detected).
        """
        lower = model_path.lower().replace("\\", "/")
        vlm_keywords = ["smolvlm", "idefics", "llava", "paligemma", "qwen2-vl",
                        "qwen2vl", "internvl", "pixtral", "medgemma", "mistral-vl",
                        "llama-3.2-11b-vision", "llama-vision"]
        if any(k in lower for k in vlm_keywords):
            return True
        # Check adapter_config.json base_model_name for adapter checkpoints
        adapter_cfg = Path(model_path) / "adapter_config.json"
        if adapter_cfg.exists():
            try:
                import json as _j
                base = _j.loads(adapter_cfg.read_text()).get("base_model_name_or_path", "").lower()
                return any(k in base for k in vlm_keywords)
            except Exception:
                pass
        return False

    def load_model(
        self,
        model_path: str,
        load_in_4bit: bool = True,
        max_seq_length: int = 2048,
        hf_token: Optional[str] = None,
    ) -> None:
        """
        Load a HF model or LoRA adapter.

        Routing:
          • VLM (SmolVLM, LLaVA, Qwen2-VL…) → FastVisionModel
            FastVisionModel.for_inference applies VLM-specific kernel patches
            (image projection, cross-attention optimisations) that
            FastLanguageModel.for_inference does not cover.
          • Text / LoRA text → FastLanguageModel (unchanged)

        Attention backend (automatic via Unsloth):
          • Flash Attention 2 if available (Linux + Ampere+)
          • Xformers memory-efficient attention otherwise (Windows default)
          • Pure PyTorch SDPA as final fallback
          Calling _attn_implementation manually would CONFLICT with Unsloth's
          own kernel patches, so we let Unsloth choose.
        """
        hf_token = hf_token if hf_token and hf_token.strip() else None
        logger.info("Loading model: %s (load_in_4bit=%s)", model_path, load_in_4bit)

        # Peek at adapter_config to decide which Unsloth loader to use
        _is_vlm_path = self._detect_vlm_path(model_path)

        if _is_vlm_path:
            try:
                from unsloth import FastVisionModel
                self.model, self.tokenizer = FastVisionModel.from_pretrained(
                    model_name=model_path,
                    max_seq_length=max_seq_length,
                    load_in_4bit=load_in_4bit,
                    token=hf_token,
                )
                FastVisionModel.for_inference(self.model)
                logger.info("Loaded via FastVisionModel (VLM path)")
            except Exception as e:
                logger.warning("FastVisionModel failed (%s), falling back to FastLanguageModel", e)
                _is_vlm_path = False  # fall through to text path below

        if not _is_vlm_path:
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                token=hf_token,
            )
            FastLanguageModel.for_inference(self.model)

        # Detect LoRA adapter (has adapter_config.json)
        self.is_lora = (Path(model_path) / "adapter_config.json").exists()
        self.model_path = model_path
        self.max_seq_length = max_seq_length

        # Load EdgeCraft training metadata (system prompt + inference params)
        self.training_system_prompt: str = ""
        self.training_inference_params: dict = {}
        meta_path = Path(model_path) / "edgecraft_metadata.json"
        if meta_path.exists():
            try:
                import json as _json
                meta = _json.loads(meta_path.read_text(encoding="utf-8"))
                self.training_system_prompt   = (meta.get("system_prompt") or "").strip()
                self.training_inference_params = meta.get("inference") or {}
                logger.info(
                    "EdgeCraft metadata loaded: system_prompt=%d chars, inference_params=%s",
                    len(self.training_system_prompt), list(self.training_inference_params.keys()),
                )
            except Exception as e:
                logger.warning("Could not load edgecraft_metadata.json: %s", e)

        logger.info(
            "Model loaded: %s (is_lora=%s, has_training_prompt=%s)",
            model_path, self.is_lora, bool(self.training_system_prompt),
        )

    # ── Prompt formatting ──

    def format_prompt(self, messages: list[dict], system_prompt: str = "") -> str:
        """Apply the model's chat template to produce a formatted prompt string.

        System prompt handling:
          - Base/instruct models (is_lora=False): always applied.
          - LoRA trained WITH system prompt (training_system_prompt set): applied —
            the model saw this format during training.
          - LoRA trained WITHOUT system prompt (training_system_prompt empty): skipped —
            injecting a system prompt changes the input distribution and causes
            the model to emit only EOS tokens (empty response).
        """
        msgs = [dict(m) for m in messages]

        # Apply system prompt when provided by user.
        # For LoRA models trained WITHOUT system prompt, injecting one can cause empty output.
        # However: if the user explicitly sets a system prompt in Chat Settings, they know
        # their model and we trust their intent. Only block if no prompt provided at all.
        if system_prompt and system_prompt.strip():
            with_system = [{"role": "system", "content": system_prompt}] + msgs
            try:
                return self.tokenizer.apply_chat_template(
                    with_system, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                # Template doesn't support system role — prepend to first user msg
                for m in msgs:
                    if m["role"] == "user":
                        m["content"] = system_prompt + "\n\n" + m["content"]
                        break

        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning("Chat template failed (%s), using manual format", e)
            lines = [f"{m['role']}: {m['content']}" for m in msgs]
            lines.append("assistant: ")
            return "\n".join(lines)

    # ── Kernel warmup ──

    def warmup_generation(self) -> None:
        """
        Run one dummy 1-token generation to trigger Triton/CUDA kernel JIT compilation.

        Why this matters:
          Unsloth uses custom Triton kernels for quantized matmuls, attention, and
          layer norms. On first use each kernel compiles (~1-5s each, dozens of kernels
          → 20-60s total delay on the very first query).  After compilation the kernels
          are cached in unsloth_compiled_cache/ on disk — warm-up is fast on subsequent
          server restarts (cache hit).

          By running the warmup during model load (before "loaded" is sent to the
          frontend), the user sees a slightly longer loading bar but zero delay on
          their first actual query.

        Text models:  1-token generation from "Hi"
        VLM models:   1-token generation from a 64×64 placeholder image + "Hi"
        """
        import torch
        logger.info("Kernel warmup: generating 1 token to compile Triton kernels...")
        try:
            if self.is_vlm:
                from PIL import Image as PILImage
                placeholder = PILImage.new("RGB", (64, 64), (80, 80, 80))
                warmup_msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hi"}]}]
                prompt = self.tokenizer.apply_chat_template(
                    warmup_msgs, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(
                    text=[prompt], images=[[placeholder]], return_tensors="pt"
                ).to(self.model.device)
            else:
                warmup_msgs = [{"role": "user", "content": "Hi"}]
                prompt = self.format_prompt(warmup_msgs)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            raw_tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            eos_id  = getattr(raw_tok, "eos_token_id", None) or getattr(self.tokenizer, "eos_token_id", None)
            pad_id  = getattr(raw_tok, "pad_token_id", None) or eos_id

            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
            logger.info("Kernel warmup complete — subsequent queries will not have compilation delay")
        except Exception as e:
            # Non-fatal: warmup failure just means the first user query will be slow
            logger.warning("Kernel warmup failed (non-fatal, first query may be slow): %s", e)

    # ── VLM helpers ──

    @property
    def is_vlm(self) -> bool:
        """True when the loaded processor has an image_processor (SmolVLM, LLaVA, Qwen2-VL…)."""
        return self.tokenizer is not None and hasattr(self.tokenizer, "image_processor")

    def _extract_image_from_messages(self, messages: list[dict]):
        """
        Scan messages for OpenAI-style image_url content and return the first PIL image found.
        Returns None if no image is present.
        """
        import base64, io
        from PIL import Image as PILImage

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:image"):
                        try:
                            header, b64 = url.split(",", 1)
                            mime = header.split(";")[0].split(":")[1]   # e.g. image/jpeg
                            img = PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                            logger.info("Image extracted from message: format=%s size=%s", mime, img.size)
                            return img
                        except Exception as e:
                            logger.error(
                                "Failed to decode base64 image (url prefix=%r): %s",
                                url[:40], e, exc_info=True,
                            )
                    else:
                        logger.warning("image_url content found but URL format unrecognised (prefix=%r)", url[:40])
        return None

    def _build_vlm_inputs(self, messages: list[dict], system_prompt: str = ""):
        """
        Build processor inputs for VLM (SmolVLM / Idefics3) inference.

        Strategy:
          • Extract image from messages (OpenAI image_url format).
          • If no image: fall back to a neutral grey placeholder so the processor
            doesn't crash — SmolVLM's Idefics3Processor always requires images.
          • Normalise message content to VLM format:
              first user turn with an image → [{"type":"image"}, {"type":"text","text":"..."}]
              all other turns              → [{"type":"text","text":"..."}]
          • Apply chat template then call processor(text=..., images=...).
        """
        from PIL import Image as PILImage

        # Extract image
        image = self._extract_image_from_messages(messages)
        has_real_image = image is not None
        if image is None:
            logger.warning("No image found in messages — using grey placeholder (model output will be uninformative)")
            image = PILImage.new("RGB", (64, 64), color=(80, 80, 80))
        else:
            logger.info("Using real image for VLM inference: size=%s", image.size)

        # Normalise messages to VLM format.
        # CRITICAL: insert {"type":"image"} in the message that CONTAINS the image_url,
        # NOT always in the first user turn.
        #
        # Why this matters for multi-turn single mode:
        #   Turn 1: user asks about burned panel (image_url present) → <image> token here ✓
        #   Turn 3: user uploads new snow panel (image_url present) → <image> token here ✓
        #   If we always put the token in turn 1, the new image is wrongly associated
        #   with turn 1's old question, causing the model to answer using old context.
        vlm_messages = []
        image_inserted = False
        fallback_user_idx: int = -1   # index of first user turn (fallback if no image found)

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            if role == "system":
                continue
            content = msg.get("content", "")

            # Check if THIS specific message contains an image
            msg_has_image = isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") in ("image", "image_url")
                for p in content
            )

            # Extract plain text from multimodal content
            if isinstance(content, list):
                text = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
            else:
                text = str(content).strip()

            # Apply system prompt into first user message
            if role == "user" and system_prompt and system_prompt.strip() and not vlm_messages:
                text = f"{system_prompt}\n\n{text}" if text else system_prompt

            # Insert image token in the message that has the image.
            # Fallback: if no message explicitly has image_url (placeholder case),
            # insert in the LAST user message (current turn) for correct context.
            if role == "user" and fallback_user_idx == -1:
                fallback_user_idx = len(vlm_messages)

            if role == "user" and msg_has_image and not image_inserted:
                vlm_messages.append({
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": text}],
                })
                image_inserted = True
            else:
                vlm_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": text}],
                })

        # Fallback: no message had image_url (placeholder case) → insert image token
        # in the LAST user message so the current question is associated with the image.
        if not image_inserted and vlm_messages:
            last_user_idx = max(
                (j for j, m in enumerate(vlm_messages) if m.get("role") == "user"),
                default=-1,
            )
            if last_user_idx >= 0:
                old = vlm_messages[last_user_idx]["content"]
                old_text = old[0]["text"] if isinstance(old, list) else str(old)
                vlm_messages[last_user_idx] = {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": old_text}],
                }
                image_inserted = True

        if not vlm_messages:
            vlm_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]

        # Apply chat template via the processor
        prompt = self.tokenizer.apply_chat_template(
            vlm_messages, tokenize=False, add_generation_prompt=True,
        )
        logger.info("VLM prompt (first 400 chars): %r", prompt[:400])

        # Processor call — images must be a list of lists (batch × images_per_sample)
        inputs = self.tokenizer(
            text=[prompt],
            images=[[image]],
            return_tensors="pt",
        ).to(self.model.device)

        return inputs

    # ── Generation ──

    def generate_stream(
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
        **kwargs,   # absorb legacy do_sample if passed
    ) -> Generator[str, None, None]:
        """
        Stream tokens from the model.

        use_adapter:
            None  → normal generation (adapters on if LoRA loaded)
            False → disable adapter layers → base model output
            True  → enable adapter layers  → LoRA output (re-enable if disabled)
        """
        from transformers import TextIteratorStreamer

        # ── VLM path (SmolVLM / LLaVA / Qwen2-VL etc.) ──
        if self.is_vlm:
            inputs = self._build_vlm_inputs(messages, system_prompt)
        else:
            # ── Text-only path ──
            prompt = self.format_prompt(messages, system_prompt)
            logger.info("Prompt (first 400 chars): %r", prompt[:400])
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # TextIteratorStreamer needs the plain tokenizer (not the whole processor)
        raw_tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        logger.info("Tokenizer type: %s  raw_tok type: %s", type(self.tokenizer).__name__, type(raw_tok).__name__)
        streamer = TextIteratorStreamer(
            raw_tok,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=0.2,
        )

        # Mirror unsloth-main exactly:
        #   do_sample = temperature > 0  (temperature=0 → greedy, >0 → sampling)
        #   eos_token_id = tokenizer.eos_token_id  (single value, not overriding list)
        #   All sampling params always passed (transformers handles them correctly)
        # For VLM models, eos/pad token ids live on the inner text tokenizer
        eos_id  = getattr(raw_tok, "eos_token_id",  None) or getattr(self.tokenizer, "eos_token_id",  None)
        pad_id  = getattr(raw_tok, "pad_token_id",  None) or getattr(self.tokenizer, "pad_token_id",  None) or eos_id

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

        logger.info(
            "Generating: do_sample=%s temperature=%.2f max_new_tokens=%d prompt_tokens=%d",
            temperature > 0, temperature, max_new_tokens, inputs["input_ids"].shape[-1],
        )

        if cancel_event is not None:
            from transformers.generation.stopping_criteria import (
                StoppingCriteria, StoppingCriteriaList,
            )

            class _CancelCriteria(StoppingCriteria):
                def __init__(self, ev):
                    self.ev = ev
                def __call__(self, input_ids, scores, **kw):
                    return self.ev.is_set()

            generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [_CancelCriteria(cancel_event)]
            )

        err: dict = {}

        def generate_fn():
            with self._generation_lock:
                try:
                    # ── Adapter toggle for compare mode ──
                    has_adapters = hasattr(self.model, "base_model")
                    if use_adapter is False and has_adapters:
                        self.model.base_model.disable_adapter_layers()
                    elif use_adapter is True and has_adapters:
                        self.model.base_model.enable_adapter_layers()

                    logger.info("Generation thread: calling model.generate()")
                    self.model.generate(**generation_kwargs)
                    logger.info("Generation thread: model.generate() returned")
                except Exception as e:
                    err["msg"] = str(e)
                    logger.error("Generation error: %s", e, exc_info=True)
                finally:
                    try:
                        streamer.end()
                    except Exception:
                        pass
                    # Always re-enable adapters so next request uses LoRA by default
                    if use_adapter is False and hasattr(self.model, "base_model"):
                        try:
                            self.model.base_model.enable_adapter_layers()
                        except Exception:
                            pass

        thread = threading.Thread(target=generate_fn, daemon=True)
        thread.start()

        from queue import Empty
        generation_complete = False
        token_count = 0
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    logger.info("Generation cancelled after %d tokens", token_count)
                    break
                try:
                    token = next(streamer)
                except StopIteration:
                    generation_complete = True
                    logger.info("Streamer StopIteration — generation done, %d tokens yielded", token_count)
                    break
                except Empty:
                    if not thread.is_alive():
                        generation_complete = True
                        logger.info("Thread died (Empty), %d tokens yielded so far", token_count)
                        break
                    continue
                if token:
                    token_count += 1
                    if token_count <= 5:
                        logger.info("Token #%d: %r", token_count, token)
                    yield token
                else:
                    logger.debug("Streamer yielded empty string (filtered)")
        finally:
            thread.join(timeout=5.0)
            logger.info("generate_stream done: %d tokens, error=%r", token_count, err.get("msg"))
            # Always raise if generation errored — even if streamer ended "cleanly"
            if err.get("msg"):
                raise RuntimeError(err["msg"])

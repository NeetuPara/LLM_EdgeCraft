"""
Training subprocess entry point.
Uses the installed `unsloth` package directly (via pip install -e unsloth-main).
No namespace conflicts with studio backend modules.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _setup_subprocess_paths() -> None:
    """Add our backend to sys.path for utils/storage access."""
    our_backend = str(Path(__file__).resolve().parent.parent.parent)
    if our_backend not in sys.path:
        sys.path.insert(0, our_backend)


def _send_status(event_queue: Any, message: str) -> None:
    event_queue.put({"type": "status", "message": message, "ts": time.time()})


def _save_edgecraft_metadata(output_dir: str, config: dict, model_name: str) -> None:
    """
    Save training metadata alongside the adapter.
    Chat sandbox reads this on model load to pre-populate:
      - system_prompt  → Settings system prompt field
      - inference.*    → temperature, max_tokens, repetition_penalty fields
    """
    import json as _json

    # Use same primary signal as _is_vlm_model — is_dataset_image is authoritative
    is_vlm = bool(config.get("is_dataset_image"))
    meta = {
        "base_model":            model_name,
        "system_prompt":         (config.get("system_prompt") or "").strip(),
        "custom_format_mapping": config.get("custom_format_mapping") or {},
        "is_vlm":                is_vlm,
        "image_column":          config.get("image_column") or "",
        # Suggested inference params — pre-populated in chat Settings
        "inference": {
            "temperature":        0,     # greedy by default for fine-tuned models
            "max_tokens":         512,
            "repetition_penalty": 1.1,
            "top_p":              0.9,
            "top_k":              40,
        },
    }
    path = Path(output_dir) / "edgecraft_metadata.json"
    try:
        path.write_text(_json.dumps(meta, indent=2, ensure_ascii=False))
        logger.info("EdgeCraft metadata saved to %s", path)
    except Exception as e:
        logger.warning("Could not save edgecraft_metadata.json: %s", e)


def _is_vlm_model(config: dict) -> bool:
    """Return True when the training job is for a Vision Language Model.

    Primary signal: is_dataset_image=True (set by frontend when modelType==='vision').
    image_column alone is NOT enough — it could be a stale value from a previous VLM
    session persisted in the frontend store.
    """
    # Primary: explicit flag set by frontend when user selected Vision model type
    if config.get("is_dataset_image"):
        return True
    # Secondary: both image column AND VLM keyword in model name must match
    has_image_col = bool(config.get("image_column"))
    lower = (config.get("model_name") or "").lower()
    vlm_keywords = ["vlm", "smolvlm", "idefics", "llava", "paligemma",
                    "internvl", "moondream", "blip", "qwen2-vl", "qwen2.5-vl"]
    name_is_vlm = any(k in lower for k in vlm_keywords)
    return has_image_col and name_is_vlm


def _load_dataset_from_config(config: dict, hf_token):
    """Load train dataset from HF or local files/folders (shared by text and VLM paths)."""
    from datasets import load_dataset
    hf_dataset    = config.get("hf_dataset", "")
    local_datasets = config.get("local_datasets")
    train_split   = config.get("train_split", "train")

    if hf_dataset and hf_dataset.strip():
        load_kwargs = {"path": hf_dataset.strip(), "split": train_split}
        subset = config.get("subset")
        if subset:
            load_kwargs["name"] = subset
        if hf_token:
            load_kwargs["token"] = hf_token
        return load_dataset(**load_kwargs)

    if local_datasets:
        all_files = []
        for fp in local_datasets:
            p = Path(fp)
            if p.is_dir():
                # Zip-extracted folder — rglob all data files
                for ext in ("*.parquet", "*.json", "*.jsonl", "*.csv"):
                    all_files.extend(str(f) for f in sorted(p.rglob(ext)))
            elif p.exists():
                all_files.append(str(p))

        if not all_files:
            return None

        SEP = "─" * 72
        ext = Path(all_files[0]).suffix.lower()
        loader = {".json": "json", ".jsonl": "json", ".csv": "csv",
                  ".parquet": "parquet"}.get(ext, "json")

        # ── DIAGNOSTIC: log every file being loaded ──
        logger.info("\n%s\n[DATASET FILE LOADING]\n  Found %d file(s) to load:", SEP, len(all_files))
        for i, f in enumerate(all_files):
            size_mb = Path(f).stat().st_size / 1024 / 1024
            logger.info("  [%d] %s  (%.1f MB)", i + 1, f, size_mb)

        dataset = load_dataset(loader, data_files=all_files, split="train")
        logger.info("  → Combined dataset: %d total rows  columns: %s\n%s",
                    len(dataset), dataset.column_names, SEP)
        return dataset

    return None


def _resolve_dataset_base_dir(base_dir: str, sample_rel_path: str) -> str:
    """
    Resolve the correct base_dir when the zip was extracted with extra nesting.

    e.g. user zipped a folder → uploads/Solar_Panel_Dataset/Solar_Panel_Dataset/dataset/...
    but parquet paths say:                                    dataset/...

    Strategy: try the given base_dir first, then walk one level of subdirectories
    to find one where the sample path actually resolves.
    """
    if not base_dir or not sample_rel_path:
        return base_dir

    # Normalise path separators
    rel = sample_rel_path.replace("\\", os.sep).replace("/", os.sep)

    # 1. Direct match
    if os.path.isfile(os.path.join(base_dir, rel)):
        return base_dir

    # 2. Check one level of subdirectories (handles double-nesting from folder zip)
    try:
        for entry in sorted(os.scandir(base_dir), key=lambda e: e.name):
            if entry.is_dir():
                candidate = os.path.join(entry.path, rel)
                if os.path.isfile(candidate):
                    logger.info(
                        "Auto-resolved extra nesting: base_dir updated to %s", entry.path
                    )
                    return entry.path
    except Exception:
        pass

    # 3. Recursive search — find the deepest directory that contains the path prefix
    first_part = rel.split(os.sep)[0]  # e.g. "dataset"
    try:
        for root, dirs, _ in os.walk(base_dir):
            if first_part in dirs:
                candidate = os.path.join(root, rel)
                if os.path.isfile(candidate):
                    logger.info("Recursive resolve: base_dir updated to %s", root)
                    return root
    except Exception:
        pass

    logger.warning("Could not resolve base_dir for path %r — using original: %s", rel, base_dir)
    return base_dir


def _run_vlm_training(event_queue, stop_queue, config: dict) -> None:
    """
    VLM fine-tuning using FastVisionModel + transformers Trainer.
    Supports any *ForConditionalGeneration model (SmolVLM/Idefics3, Gemma3-VL, etc.)
    """
    import threading
    import queue as _queue
    from pathlib import Path as _Path

    model_name      = config["model_name"]
    hf_token        = config.get("hf_token") or None
    load_in_4bit    = config.get("load_in_4bit", True)
    use_lora        = config.get("use_lora", True)
    image_column    = config.get("image_column") or "image"
    dataset_base_dir = config.get("dataset_base_dir") or ""

    # Column mapping: input columns → question text, output columns → answer text
    col_mapping   = config.get("custom_format_mapping") or {}
    question_cols = [c for c, r in col_mapping.items() if r in ("input", "user", "instruction")]
    answer_cols   = [c for c, r in col_mapping.items() if r in ("output", "assistant")]

    training_start = time.time()

    _send_status(event_queue, "Importing Unsloth (VLM)...")
    from unsloth import FastVisionModel, is_bfloat16_supported
    from transformers import TrainingArguments, Trainer, TrainerCallback

    # Warn if base model detected — Instruct models are strongly recommended for VLM fine-tuning
    lower_name = model_name.lower()
    if "base" in lower_name and "instruct" not in lower_name:
        warn_msg = (
            f"WARNING: '{model_name}' appears to be a BASE model. "
            "Base VLMs have no instruction tuning and may produce poor results. "
            "Use the Instruct version instead (e.g. SmolVLM-500M-Instruct)."
        )
        logger.warning(warn_msg)
        _send_status(event_queue, f"⚠ {warn_msg}")

    logger.info("VLM training: model=%s load_in_4bit=%s use_lora=%s", model_name, load_in_4bit, use_lora)

    # ── Load model ──
    _send_status(event_queue, f"Loading VLM {model_name}...")
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.get("max_seq_length", 2048),
        load_in_4bit=load_in_4bit,
        full_finetuning=not use_lora,
        token=hf_token,
    )

    # ── Configure LoRA ──
    if use_lora:
        _send_status(event_queue, "Configuring LoRA adapters (VLM)...")
        gc_setting = config.get("gradient_checkpointing", "unsloth")
        if gc_setting in ("none", "", None, "false", "False"):
            gc_setting = False
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers    = config.get("finetune_vision_layers", False),
            finetune_language_layers  = config.get("finetune_language_layers", True),
            finetune_attention_modules= config.get("finetune_attention_modules", True),
            finetune_mlp_modules      = config.get("finetune_mlp_modules", True),
            r                 = config.get("lora_r", 16),
            lora_alpha        = config.get("lora_alpha", 16),
            lora_dropout      = config.get("lora_dropout", 0.0),
            bias              = "none",
            use_gradient_checkpointing = gc_setting,
            random_state      = config.get("random_seed", 3407),
            use_rslora        = config.get("use_rslora", False),
        )

    FastVisionModel.for_training(model)

    # ── Load dataset ──
    _send_status(event_queue, "Loading VLM dataset...")
    dataset = _load_dataset_from_config(config, hf_token)
    if dataset is None:
        event_queue.put({"type": "error", "error": "No dataset specified", "stack": "", "ts": time.time()})
        return
    logger.info("VLM dataset loaded: %d rows, columns: %s", len(dataset), dataset.column_names)

    # Apply slicing
    slice_start = config.get("dataset_slice_start")
    slice_end   = config.get("dataset_slice_end")
    if slice_start or slice_end:
        start = slice_start or 0
        end   = slice_end or len(dataset)
        dataset = dataset.select(range(start, min(end, len(dataset))))

    # ── Auto-resolve base_dir for double-nested zips ──
    # e.g. user zipped a folder → Solar_Panel_Dataset/Solar_Panel_Dataset/dataset/...
    try:
        first_img = dataset[0].get(image_column)
        if isinstance(first_img, dict) and first_img.get("bytes") is None:
            sample_rel = first_img.get("path") or ""
            if sample_rel and dataset_base_dir:
                dataset_base_dir = _resolve_dataset_base_dir(dataset_base_dir, sample_rel)
                logger.info("dataset_base_dir (after resolve): %s", dataset_base_dir)
    except Exception as _e:
        logger.debug("Base dir auto-resolve failed: %s", _e)

    # ── DIAGNOSTIC: validate image paths before training starts ──
    SEP = "─" * 72
    try:
        rows_to_check = list(dataset.select(range(min(3, len(dataset)))))
        logger.info("\n%s\n[VLM IMAGE PATH VALIDATION]\n  dataset_base_dir: %r\n  image_column   : %r",
                    SEP, dataset_base_dir, image_column)

        ok_count, fail_count = 0, 0
        for i, row in enumerate(rows_to_check):
            img_data = row.get(image_column)
            if isinstance(img_data, dict) and img_data.get("bytes") is None:
                rel_path = img_data.get("path") or ""
                abs_path = (
                    os.path.join(dataset_base_dir, rel_path)
                    if dataset_base_dir and not os.path.isabs(rel_path)
                    else rel_path
                )
                exists = os.path.isfile(abs_path)
                status = "✓ FOUND" if exists else "✗ MISSING"
                logger.info("  Row %d: %s\n    rel: %s\n    abs: %s", i, status, rel_path, abs_path)
                if exists: ok_count += 1
                else: fail_count += 1
            else:
                logger.info("  Row %d: image bytes present (no path resolution needed)", i)
                ok_count += 1

        if fail_count > 0:
            logger.warning("⚠ %d/%d image paths NOT FOUND. Check dataset_base_dir and zip structure.",
                           fail_count, ok_count + fail_count)
            _send_status(event_queue,
                f"⚠ WARNING: {fail_count} image paths not found on disk. "
                "Check that images are inside the zip under a 'dataset/' subfolder.")
        else:
            logger.info("✓ All %d sampled image paths resolved successfully.", ok_count)
            _send_status(event_queue, f"✓ Image paths verified ({ok_count} sampled OK)")
        logger.info(SEP)

    except Exception as _e:
        logger.warning("Image path validation failed: %s", _e)

    # Optional train/eval split
    eval_dataset = None
    eval_split = config.get("eval_split") or ""
    if eval_split and eval_split.strip():
        pass  # eval from separate split — not yet implemented for VLM
    elif len(dataset) > 20:
        split = dataset.train_test_split(test_size=0.05, seed=42)
        dataset, eval_dataset = split["train"], split["test"]

    # ── Resolve image token ids (SmolVLM/Idefics3 uses multiple image special tokens) ──
    # Must mask ALL image-related tokens so model doesn't try to predict image patches.
    # Without this, loss explodes to 15-20+ because image patch tokens dominate.
    image_token_ids: set[int] = set()

    # Get the raw tokenizer (FastVisionModel may wrap it)
    raw_tok = getattr(processor, "tokenizer", processor)
    raw_tok = getattr(raw_tok, "tokenizer", raw_tok)  # unwrap one more level if needed
    unk_id = getattr(raw_tok, "unk_token_id", -999)

    def _try_add_token(tok_str: str):
        # Method 1: additional_special_tokens list
        try:
            tok_obj = getattr(processor, "tokenizer", processor)
            idx = tok_obj.additional_special_tokens.index(tok_str)
            tid = tok_obj.additional_special_tokens_ids[idx]
            if tid not in (None, unk_id):
                image_token_ids.add(int(tid))
                return
        except (ValueError, AttributeError):
            pass
        # Method 2: convert_tokens_to_ids
        try:
            tid = raw_tok.convert_tokens_to_ids(tok_str)
            if tid not in (None, unk_id):
                image_token_ids.add(int(tid))
        except Exception:
            pass

    # SmolVLM/Idefics3 image special tokens
    for tok in ["<image>", "<fake_token_around_image>", "<global-img>",
                "<row_1_col_1>", "<|image|>"]:
        _try_add_token(tok)

    # Also check processor-level attribute
    for attr in ["image_token_id", "image_token_index"]:
        val = getattr(processor, attr, None) or getattr(raw_tok, attr, None)
        if val is not None and int(val) != unk_id:
            image_token_ids.add(int(val))

    # Collect ALL row_X_col_Y tokens (image patch grid tokens)
    try:
        vocab = raw_tok.get_vocab()
        for tok_str, tok_id in vocab.items():
            if (tok_str.startswith("<row_") or
                    tok_str in ("<fake_token_around_image>", "<global-img>", "<image>")):
                image_token_ids.add(tok_id)
    except Exception:
        pass

    if image_token_ids:
        logger.info("Image token IDs found (%d tokens) — will mask in labels. Sample: %s",
                    len(image_token_ids), sorted(image_token_ids)[:5])
    else:
        logger.warning("Could not find any image token IDs — image tokens will NOT be masked. "
                       "Expect high loss (~15+). Use SmolVLM-500M-Instruct for better results.")

    # ── Build collate_fn ──
    system_prompt = (config.get("system_prompt") or "").strip()

    def collate_fn(examples):
        from PIL import Image as PILImage
        texts, images = [], []

        for ex in examples:
            # ── Load image ──
            img_data = ex.get(image_column)
            if isinstance(img_data, dict):
                # bytes=None → resolve from path
                if img_data.get("bytes") is None:
                    rel_path = img_data.get("path") or ""
                    img_path = (
                        os.path.join(dataset_base_dir, rel_path)
                        if dataset_base_dir and not os.path.isabs(rel_path)
                        else rel_path
                    )
                    try:
                        image = PILImage.open(img_path).convert("RGB")
                    except Exception as e:
                        logger.warning("Failed to load image %r: %s — using blank", img_path, e)
                        image = PILImage.new("RGB", (224, 224), (128, 128, 128))
                else:
                    import io as _io
                    image = PILImage.open(_io.BytesIO(img_data["bytes"])).convert("RGB")
            elif hasattr(img_data, "convert"):
                image = img_data.convert("RGB") if img_data.mode != "RGB" else img_data
            else:
                logger.warning("Unknown image type %s — using blank", type(img_data))
                image = PILImage.new("RGB", (224, 224), (128, 128, 128))

            # ── Build question text ──
            question_parts = [str(ex.get(c) or "").strip() for c in question_cols if ex.get(c)]
            question = " ".join(question_parts) if question_parts else ""
            answer_parts = [str(ex.get(c) or "").strip() for c in answer_cols if ex.get(c)]
            answer = "\n\n".join(answer_parts) if answer_parts else ""

            # ── Build message with optional system prompt ──
            user_content = []
            if system_prompt:
                user_content.append({"type": "text", "text": system_prompt})
            user_content.append({"type": "image"})
            if question:
                user_content.append({"type": "text", "text": question})

            messages = [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        # Mask pad tokens
        pad_id = getattr(processor.tokenizer, "pad_token_id", None) or \
                 getattr(getattr(processor, "tokenizer", processor), "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        # Mask ALL image-related special tokens (prevents loss explosion)
        for tid in image_token_ids:
            labels[labels == tid] = -100
        batch["labels"] = labels
        return batch

    # Log first formatted example
    try:
        sample_ex = dict(next(iter(dataset)))
        q_parts  = [str(sample_ex.get(c) or "").strip() for c in question_cols]
        a_parts  = [str(sample_ex.get(c) or "").strip() for c in answer_cols]
        SEP = "─" * 72
        logger.info("\n%s\n[VLM TRAINING FORMAT — first example]\n"
                    "  question cols: %s\n  answer cols  : %s\n"
                    "  question  : %s\n  answer    : %s\n"
                    "  image col : %r (type: %s)\n%s",
                    SEP, question_cols, answer_cols,
                    " | ".join(q_parts)[:300], " | ".join(a_parts)[:300],
                    image_column, type(sample_ex.get(image_column)).__name__, SEP)
    except Exception:
        pass

    # ── Output dir ──
    from utils.paths import outputs_root
    base_dir_out = outputs_root()
    base_dir_out.mkdir(parents=True, exist_ok=True)
    custom_name = (config.get("output_dir") or "").strip()
    if custom_name:
        output_dir = str(base_dir_out / custom_name)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe = model_name.replace("/", "_").replace("\\", "_")
        output_dir = str(base_dir_out / f"{safe}_{ts}")

    # ── Learning rate ──
    lr_raw = config.get("learning_rate", "2e-5")
    try:
        lr = float(lr_raw)
    except (ValueError, TypeError):
        lr = 2e-5

    # ── Progress callback — must send "progress" type to match pump loop ──
    class _VLMProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or not state.is_local_process_zero:
                return
            step  = state.global_step
            total = state.max_steps or 1
            loss     = logs.get("loss")
            lr       = logs.get("learning_rate")
            grad_norm = logs.get("grad_norm")
            epoch    = logs.get("epoch", 0)

            # "progress" is the event type the pump loop (_handle_event) expects
            event_queue.put({
                "type":          "progress",
                "step":          step,
                "total_steps":   total,
                "epoch":         epoch,
                "loss":          loss,
                "learning_rate": lr,
                "grad_norm":     grad_norm,
                "ts":            time.time(),
            })

        def on_epoch_end(self, args, state, control, **kwargs):
            event_queue.put({
                "type":    "progress",
                "step":    state.global_step,
                "total_steps": state.max_steps or 1,
                "epoch":   state.epoch,
                "status_message": f"Epoch {int(state.epoch)} complete",
                "ts":      time.time(),
            })

        def on_train_end(self, args, state, control, **kwargs):
            event_queue.put({"type": "complete", "ts": time.time()})

    # ── Build TrainingArguments ──
    num_epochs = config.get("num_epochs", 2)
    max_steps  = config.get("max_steps", 0)

    training_args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = num_epochs if not max_steps else 1,
        max_steps                   = max_steps if max_steps else -1,
        per_device_train_batch_size = config.get("batch_size", 2),
        gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4),
        warmup_steps                = config.get("warmup_steps") or 5,
        learning_rate               = lr,
        lr_scheduler_type           = config.get("lr_scheduler_type", "cosine"),
        optim                       = config.get("optim", "adamw_8bit"),
        weight_decay                = config.get("weight_decay", 0.001),
        fp16                        = not is_bfloat16_supported(),
        bf16                        = is_bfloat16_supported(),
        logging_steps               = 10,
        seed                        = config.get("random_seed", 3407),
        report_to                   = "none",
        save_strategy               = config.get("save_strategy", "no"),
        save_steps                  = config.get("save_steps") or 250,
        remove_unused_columns       = False,   # CRITICAL: keeps image column in dataset
        dataloader_num_workers      = 0,       # Windows: cannot pickle closures across processes
        dataloader_pin_memory       = False,   # Windows: cannot pin non-CUDA tensors
        gradient_checkpointing      = True,
    )

    trainer = Trainer(
        model          = model,
        args           = training_args,
        data_collator  = collate_fn,
        train_dataset  = dataset,
        eval_dataset   = eval_dataset,
        callbacks      = [_VLMProgressCallback()],
    )

    _send_status(event_queue, "VLM training started...")
    trainer.train()

    # ── Save ──
    _send_status(event_queue, "Saving VLM adapter...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    _save_edgecraft_metadata(output_dir, config, model_name)
    logger.info("VLM model saved to %s", output_dir)

    duration = time.time() - training_start
    event_queue.put({
        "type": "complete",
        "output_dir": output_dir,
        "duration_seconds": round(duration),
        "ts": time.time(),
    })


def run_training_process(
    *,
    event_queue: Any,
    stop_queue: Any,
    config: dict,
) -> None:
    """Subprocess entry point. Uses unsloth package directly."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = "ignore"

    import warnings
    warnings.filterwarnings("ignore")

    _setup_subprocess_paths()

    # ── Compiled cache MUST be on PYTHONPATH before any subprocess spawns ──
    # Unsloth compiles SFTTrainer variants into unsloth_compiled_cache/.
    # When datasets.map() spawns worker processes (even num_proc=1 on Windows),
    # they need to find UnslothSFTTrainer. We propagate the cache path via
    # PYTHONPATH so all child processes inherit it — mirrors unsloth-main trainer.py.
    _compile_cache = os.environ.get("UNSLOTH_COMPILE_LOCATION", "")
    if not _compile_cache:
        # Anchor to backend/ directory (parent of core/training/worker.py),
        # not CWD — so it's correct regardless of where run.py was launched from.
        _backend_dir = str(Path(__file__).resolve().parent.parent.parent)
        _compile_cache = os.path.join(_backend_dir, "unsloth_compiled_cache")
    elif not os.path.isabs(_compile_cache):
        _compile_cache = os.path.abspath(_compile_cache)
    os.environ["UNSLOTH_COMPILE_LOCATION"] = _compile_cache
    _pp = os.environ.get("PYTHONPATH", "")
    if _compile_cache not in _pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = _compile_cache + (os.pathsep + _pp if _pp else "")
    if _compile_cache not in sys.path:
        sys.path.insert(0, _compile_cache)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Apply GPU selection
    from utils.hardware import apply_gpu_ids
    apply_gpu_ids(config.get("resolved_gpu_ids"))

    # On Windows: check Triton
    if sys.platform == "win32":
        try:
            import triton  # noqa: F401
        except ImportError:
            os.environ["TORCHDYNAMO_DISABLE"] = "1"

    # On Linux: set fork for dataset.map() parallelism
    if sys.platform == "linux":
        import multiprocessing as _mp
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass

    model_name = config["model_name"]
    training_start = time.time()

    # ── Branch: VLM vs text ──
    if _is_vlm_model(config):
        try:
            _run_vlm_training(event_queue, stop_queue, config)
        except Exception as exc:
            logger.error("VLM training failed: %s", exc, exc_info=True)
            event_queue.put({"type": "error", "error": str(exc),
                             "stack": traceback.format_exc(), "ts": time.time()})
        return

    try:
        _send_status(event_queue, "Importing Unsloth...")

        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer, SFTConfig
        from transformers import TrainerCallback
        from datasets import load_dataset
        import threading
        import queue as _queue

        logger.info("Subprocess imports complete — unsloth loaded")

        # ── Load model ──
        _send_status(event_queue, f"Loading model {model_name}...")
        hf_token = config.get("hf_token") or None
        if hf_token and not hf_token.strip():
            hf_token = None

        load_in_4bit = config.get("load_in_4bit", True)
        use_lora = config.get("use_lora", True)
        full_finetuning = not use_lora

        training_mode = "QLoRA (4-bit NF4)" if (use_lora and load_in_4bit) else \
                        "LoRA (16-bit bfloat16)" if use_lora else "Full Fine-tune"
        logger.info("Training mode: %s | load_in_4bit=%s | use_lora=%s",
                    training_mode, load_in_4bit, use_lora)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=config.get("max_seq_length", 2048),
            load_in_4bit=load_in_4bit,
            full_finetuning=full_finetuning,
            token=hf_token,
        )

        # ── Configure LoRA ──
        if use_lora:
            _send_status(event_queue, "Configuring LoRA adapters...")
            target_modules_raw = config.get("target_modules")
            if isinstance(target_modules_raw, str) and target_modules_raw.strip():
                target_modules = [m.strip() for m in target_modules_raw.split(",") if m.strip()]
            else:
                target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]

            gc_setting = config.get("gradient_checkpointing", "unsloth")
            if gc_setting in ("none", "", None, "false", "False"):
                gc_setting = False

            model = FastLanguageModel.get_peft_model(
                model,
                r=config.get("lora_r", 64),
                target_modules=target_modules,
                lora_alpha=config.get("lora_alpha", 16),
                lora_dropout=config.get("lora_dropout", 0.0),
                bias="none",
                use_gradient_checkpointing=gc_setting,
                random_state=config.get("random_seed", 3407),
                use_rslora=config.get("use_rslora", False),
            )

        # ── Load dataset ──
        _send_status(event_queue, "Loading dataset...")
        dataset = _load_dataset_from_config(config, hf_token)
        if dataset is not None:
            _send_status(event_queue, f"Dataset loaded: {len(dataset)} samples")

        if dataset is None:
            event_queue.put({"type": "error", "error": "No dataset specified (hf_dataset and local_datasets are both empty)", "stack": "", "ts": time.time()})
            return

        # Apply dataset slicing
        slice_start = config.get("dataset_slice_start")
        slice_end = config.get("dataset_slice_end")
        if slice_start is not None or slice_end is not None:
            start = slice_start or 0
            end = slice_end or len(dataset)
            dataset = dataset.select(range(start, min(end, len(dataset))))

        # ── Format dataset into "text" column ──
        _send_status(event_queue, "Formatting dataset...")
        original_fmt = _detect_dataset_format(dataset)   # capture BEFORE formatting
        dataset = _format_dataset_for_training(dataset, tokenizer, model_name, config)
        logger.info("Dataset formatted — columns: %s, rows: %d",
                     dataset.column_names, len(dataset))

        # ── Keep only "text" column ──
        # SFTTrainer requires ONLY the "text" column (or "input_ids" if pre-tokenized).
        # Any extra columns confuse the data collator:
        #   - "labels" → reserved name, expects a loss tensor not a string
        #   - "explanation", "id", etc. → cause collation errors
        # User's metadata columns (like 'labels': 'fair/unfair') are NOT training signals.
        extra_cols = [c for c in dataset.column_names if c != "text"]
        if extra_cols:
            logger.info("Removing extra columns before training: %s", extra_cols)
            dataset = dataset.remove_columns(extra_cols)

        # ── Auto-split 10% for eval if no explicit eval dataset configured ──
        # This ensures eval loss always shows on the training chart.
        # Only skip if user explicitly set eval_split to a separate HF split.
        eval_dataset = None
        explicit_eval_split = (config.get("eval_split") or "").strip()

        if explicit_eval_split:
            # User specified a separate HF split — load it
            _send_status(event_queue, f"Loading eval split: {explicit_eval_split}...")
            try:
                eval_hf = config.get("hf_dataset", "")
                if eval_hf:
                    eval_raw = load_dataset(eval_hf.strip(), split=explicit_eval_split,
                                            token=hf_token)
                    eval_dataset = _format_dataset_for_training(eval_raw, tokenizer, model_name, config)
                    # Remove extra columns from eval too
                    eval_extra = [c for c in eval_dataset.column_names if c != "text"]
                    if eval_extra:
                        eval_dataset = eval_dataset.remove_columns(eval_extra)
                    logger.info("Eval split '%s' loaded: %d rows", explicit_eval_split, len(eval_dataset))
            except Exception as e:
                logger.warning("Could not load eval split '%s': %s — skipping eval", explicit_eval_split, e)
                eval_dataset = None
        else:
            # No explicit eval split — auto-split 10% from training data
            if len(dataset) >= 20:  # only split if dataset is large enough
                split = dataset.train_test_split(
                    test_size=0.1,
                    seed=config.get("random_seed", 3407),
                )
                dataset = split["train"]
                eval_dataset = split["test"]
                logger.info("Auto-split: %d train, %d eval (10%%)", len(dataset), len(eval_dataset))
                _send_status(event_queue, f"Auto-split: {len(dataset)} train / {len(eval_dataset)} eval (10%)")
                # Notify frontend that eval is now configured
                event_queue.put({"type": "eval_configured", "ts": time.time()})
            else:
                logger.info("Dataset too small for auto-split (%d rows) — skipping eval", len(dataset))

        # ── Stop signal handler ──
        _should_stop = False
        _save_on_stop = True

        def _poll_stop():
            nonlocal _should_stop, _save_on_stop
            while True:
                try:
                    msg = stop_queue.get(timeout=1.0)
                    if msg and msg.get("type") == "stop":
                        _should_stop = True
                        _save_on_stop = msg.get("save", True)
                        logger.info("Stop signal received (save=%s)", _save_on_stop)
                        return
                except _queue.Empty:
                    continue
                except (EOFError, OSError):
                    return

        threading.Thread(target=_poll_stop, daemon=True).start()

        # ── Progress callback ──
        class _ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                elapsed = time.time() - training_start
                step = state.global_step
                # Use trainer's max_steps once available, else fall back to pre-computed estimate
                total = (state.max_steps if state.max_steps and state.max_steps > 0
                         else _estimated_total_steps)
                eta = ((elapsed / step) * (total - step)) if step > 0 and total > step else None
                event_queue.put({
                    "type": "progress",
                    "step": step,
                    "epoch": round(state.epoch or 0, 2),
                    "loss": logs.get("loss"),
                    "learning_rate": logs.get("learning_rate"),
                    "total_steps": total,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "grad_norm": logs.get("grad_norm"),
                    "eval_loss": logs.get("eval_loss"),
                    "num_tokens": getattr(state, "num_input_tokens_seen", None),
                    "status_message": f"Training step {step}/{total}",
                    "ts": time.time(),
                })

            def on_step_end(self, args, state, control, **kwargs):
                if _should_stop:
                    control.should_training_stop = True
                return control

        # ── Output directory ──
        import re as _re
        custom_name = (config.get("output_dir") or "").strip()
        if custom_name:
            output_dir = _re.sub(r"[^\w\-.]", "_", custom_name).strip("_.")
        else:
            output_dir = f"{model_name.split('/')[-1]}_{int(time.time())}"
        from utils.paths import resolve_output_dir, ensure_dir
        output_dir = str(resolve_output_dir(output_dir))
        ensure_dir(Path(output_dir))

        # ── Training arguments ──
        lr = float(config.get("learning_rate", "2e-4"))
        num_epochs = config.get("num_epochs", 3)
        max_steps = config.get("max_steps", 0)

        # Pre-compute estimated total steps for accurate progress bar from step 1
        _batch_size = config.get("batch_size", 2)
        _grad_accum = config.get("gradient_accumulation_steps", 4)
        _effective_batch = max(_batch_size * _grad_accum, 1)
        _epochs = num_epochs if not (max_steps and int(max_steps) > 0) else 1
        _estimated_total_steps = (
            int(max_steps) if (max_steps and int(max_steps) > 0)
            else max(1, (len(dataset) // _effective_batch) * _epochs)
        )

        sft_kwargs: dict = {
            "output_dir": output_dir,
            "per_device_train_batch_size": config.get("batch_size", 2),
            "per_device_eval_batch_size": config.get("batch_size", 2),
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 4),
            "learning_rate": lr,
            "lr_scheduler_type": config.get("lr_scheduler_type", "linear"),
            "optim": config.get("optim", "adamw_8bit"),
            "weight_decay": config.get("weight_decay", 0.001),
            "warmup_steps": config.get("warmup_steps") or 5,
            "fp16": not is_bfloat16_supported(),
            "bf16": is_bfloat16_supported(),
            "logging_steps": 10,
            "seed": config.get("random_seed", 3407),
            "report_to": "none",
            "max_seq_length": config.get("max_seq_length", 2048),
            "dataset_text_field": "text",   # explicitly tell SFTTrainer which column to use
            # dataset_num_proc: NOT set — let SFTTrainer use its default.
            # The compiled cache is on PYTHONPATH (set above) so subprocesses can find UnslothSFTTrainer.
        }

        if max_steps and int(max_steps) > 0:
            sft_kwargs["max_steps"] = int(max_steps)
        else:
            sft_kwargs["num_train_epochs"] = num_epochs

        save_steps = config.get("save_steps", 0)
        save_strategy = config.get("save_strategy", "no")
        if save_strategy == "best":
            # Best epoch: save every epoch, keep only the one with lowest eval loss
            sft_kwargs["save_strategy"] = "epoch"
            sft_kwargs["load_best_model_at_end"] = True
            sft_kwargs["metric_for_best_model"] = "eval_loss"
            sft_kwargs["greater_is_better"] = False
            sft_kwargs["save_total_limit"] = 1        # keep only best checkpoint on disk
        elif save_strategy == "epoch":
            sft_kwargs["save_strategy"] = "epoch"
            sft_kwargs["load_best_model_at_end"] = False
        elif save_steps and int(save_steps) > 0:
            sft_kwargs["save_steps"] = int(save_steps)
            sft_kwargs["save_strategy"] = "steps"
        else:
            sft_kwargs["save_strategy"] = "no"    # save only at end via model.save_pretrained()

        # WandB
        if config.get("enable_wandb"):
            wandb_token = config.get("wandb_token")
            if wandb_token:
                os.environ["WANDB_API_KEY"] = wandb_token
            os.environ["WANDB_PROJECT"] = config.get("wandb_project", "unslothcraft")
            sft_kwargs["report_to"] = "wandb"

        # TensorBoard
        if config.get("enable_tensorboard"):
            tb_dir = config.get("tensorboard_dir", "runs")
            from utils.paths import base_root
            tb_path = base_root() / tb_dir
            ensure_dir(tb_path)
            sft_kwargs["logging_dir"] = str(tb_path)
            if sft_kwargs["report_to"] == "none":
                sft_kwargs["report_to"] = "tensorboard"
            else:
                sft_kwargs["report_to"] = [sft_kwargs["report_to"], "tensorboard"]

        # Add eval settings when eval dataset is available
        if eval_dataset is not None:
            if save_strategy == "best":
                # Early stopping requires eval_strategy to match save_strategy (both "epoch")
                sft_kwargs["eval_strategy"] = "epoch"
            else:
                # Evaluate every 20% of training (5 eval points per run)
                total_steps_est = max(1, len(dataset) // max(config.get("batch_size", 2) * config.get("gradient_accumulation_steps", 4), 1))
                eval_every = max(10, total_steps_est // 5)
                sft_kwargs["eval_strategy"] = "steps"
                sft_kwargs["eval_steps"] = eval_every

        training_args = SFTConfig(**sft_kwargs)

        eval_info = f", eval={len(eval_dataset)} rows" if eval_dataset is not None else ", no eval"
        _send_status(event_queue, f"Starting training on {model_name}...")
        dataset_ref = config.get("hf_dataset") or config.get("local_datasets")
        logger.info("Training config: epochs=%s, lr=%s, batch=%s, lora_r=%s, dataset=%s (%d rows%s)",
                     num_epochs, lr, config.get("batch_size"), config.get("lora_r"),
                     dataset_ref, len(dataset), eval_info)

        callbacks = [_ProgressCallback()]
        if save_strategy == "best" and eval_dataset is not None:
            from transformers import EarlyStoppingCallback
            patience = config.get("early_stopping_patience", 3)
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
            logger.info("Early stopping enabled — patience=%d epochs", patience)
        elif save_strategy == "best" and eval_dataset is None:
            logger.warning("save_strategy='best' requires eval data — no eval dataset found, "
                           "falling back to saving every epoch without early stopping")

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            callbacks=callbacks,
        )

        # Apply train_on_completions_only masking if configured
        # Only applies to instruction/chat datasets (not raw text datasets)
        if config.get("train_on_completions", True) and original_fmt != "text":
            trainer = _apply_train_on_completions(trainer, tokenizer, model_name)

        training_complete = False
        try:
            trainer.train()
            training_complete = True
        except Exception as train_exc:
            # Check if training actually finished all steps despite the exception
            # (common on Windows: checkpoint save fails after last step)
            try:
                state = trainer.state
                finished = (state.global_step >= state.max_steps > 0) or \
                           (state.epoch is not None and state.epoch >= (config.get("num_epochs", 1) - 0.01))
            except Exception:
                finished = False

            if finished:
                logger.warning("trainer.train() raised after completing all steps — "
                               "likely a checkpoint write error. Proceeding to save model. Error: %s", train_exc)
                training_complete = True
            else:
                raise  # re-raise if training genuinely failed mid-run

        # ── Save model ──
        if _should_stop and not _save_on_stop:
            event_queue.put({"type": "complete", "output_dir": None, "status_message": "Training cancelled", "ts": time.time()})
            return

        _send_status(event_queue, "Saving model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        _save_edgecraft_metadata(output_dir, config, model_name)
        logger.info("Model saved to %s", output_dir)

        event_queue.put({
            "type": "complete",
            "output_dir": output_dir,
            "status_message": "Training completed! Model saved.",
            "ts": time.time(),
        })

    except Exception as exc:
        logger.error("Training error: %s", exc, exc_info=True)
        event_queue.put({
            "type": "error",
            "error": str(exc),
            "stack": traceback.format_exc(limit=20),
            "ts": time.time(),
        })



# ═══════════════════════════════════════════════════════════════════
# Dataset formatting — ported from unsloth-main logic
# Stage 1: detect + standardize format
# Stage 2: apply model chat template → "text" column with EOS token
# Stage 3: train_on_responses_only masking (applied after SFTTrainer)
# ═══════════════════════════════════════════════════════════════════

# Model name substring → unsloth template name (most-specific first)
_MODEL_TEMPLATE_MAP = [
    ("llama-3",    "llama-3.1"),
    ("llama3",     "llama-3.1"),
    ("qwen3",      "qwen3"),
    ("qwen-3",     "qwen3"),
    ("qwen2.5-vl", "qwen-2.5"),
    ("qwen2.5",    "qwen-2.5"),
    ("qwen2-vl",   "qwen-2.5"),
    ("qwen2",      "qwen-2.5"),
    ("qwen",       "qwen-2.5"),
    ("gemma-3n",   "gemma-3n"),
    ("gemma3n",    "gemma-3n"),
    ("gemma-3",    "gemma-3"),
    ("gemma3",     "gemma-3"),
    ("gemma",      "gemma"),
    ("mistral",    "mistral"),
    ("mixtral",    "mistral"),
    ("phi-4",      "phi-4"),
    ("phi-3.5",    "phi-3.5"),
    ("phi-3",      "phi-3"),
    ("phi",        "phi-4"),
    ("deepseek-r1", "qwen-2.5"),
    ("deepseek",   "chatml"),
    ("smollm",     "chatml"),
    ("smolvlm",    "chatml"),
]

_ROLE_ALIASES = {
    "human": "user", "input": "user",
    "gpt": "assistant", "bot": "assistant", "output": "assistant",
    "system": "system",
}


def _get_template_name(model_name: str):
    lower = model_name.lower()
    for substr, tpl in _MODEL_TEMPLATE_MAP:
        if substr in lower:
            return tpl
    return None


def _setup_tokenizer_template(model_name: str, tokenizer):
    """
    Apply chat template to tokenizer.
    Priority: 1) unsloth mapper  2) tokenizer's own  3) ChatML fallback
    Mirrors unsloth-main's get_tokenizer_chat_template().
    """
    try:
        from unsloth.chat_templates import get_chat_template
        tpl = _get_template_name(model_name)
        if tpl:
            tokenizer = get_chat_template(tokenizer, chat_template=tpl)
            logger.info("Chat template: %s (model: %s)", tpl, model_name)
        elif getattr(tokenizer, "chat_template", None):
            logger.info("Using tokenizer's own chat template")
        else:
            # Base model — no template → apply ChatML
            tokenizer = get_chat_template(tokenizer, chat_template="chatml")
            logger.info("No chat template found — applied ChatML fallback")
    except Exception as e:
        logger.warning("Could not apply chat template: %s", e)
    return tokenizer


def _detect_dataset_format(dataset) -> str:
    cols = set(c.lower() for c in dataset.column_names)
    if "text" in cols:
        return "text"
    if "conversations" in cols or "messages" in cols:
        return "sharegpt"
    if "instruction" in cols and ("output" in cols or "response" in cols):
        return "alpaca"
    if ("question" in cols or "query" in cols) and ("answer" in cols or "response" in cols):
        return "qa"
    if ("prompt" in cols or "input" in cols) and ("completion" in cols or "output" in cols):
        return "completion"
    return "unknown"


def _get_original_column(dataset, *candidates):
    col_map = {c.lower(): c for c in dataset.column_names}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None


def _nproc():
    """1 on Windows/macOS (spawn), None on Linux."""
    return 1 if sys.platform in ("win32", "darwin") else None


def _apply_custom_mapping(dataset, mapping: dict):
    """
    Apply user-defined column mapping and produce standard alpaca columns.

    mapping format: {original_column: role}
    e.g. {"text": "input", "labels": "output", "explanation": "output"}

    Supported roles (case-insensitive):
      input / user / instruction / human  → user turn content
      output / assistant / gpt            → assistant turn content

    Multiple columns mapped to the same role are concatenated with "\\n\\n".
    Result: dataset with exactly two columns — "instruction" (user) + "output" (assistant).
    """
    _INPUT_ROLES  = {"input", "user", "instruction", "human"}
    _OUTPUT_ROLES = {"output", "assistant", "gpt"}

    input_cols  = [c for c, r in mapping.items() if r.lower() in _INPUT_ROLES  and c in dataset.column_names]
    output_cols = [c for c, r in mapping.items() if r.lower() in _OUTPUT_ROLES and c in dataset.column_names]

    if not input_cols and not output_cols:
        return dataset

    def _merge(examples):
        result = {}
        if input_cols:
            n = len(examples[input_cols[0]])
            result["instruction"] = [
                "\n\n".join(str(examples[c][i] or "").strip() for c in input_cols if str(examples[c][i] or "").strip())
                for i in range(n)
            ]
        if output_cols:
            n = len(examples[output_cols[0]])
            result["output"] = [
                "\n\n".join(str(examples[c][i] or "").strip() for c in output_cols if str(examples[c][i] or "").strip())
                for i in range(n)
            ]
        return result

    orig_cols = dataset.column_names
    dataset = dataset.map(_merge, batched=True, num_proc=_nproc())
    # Drop all original columns (merged ones are already added by map)
    to_drop = [c for c in orig_cols if c in dataset.column_names and c not in ("instruction", "output")]
    if to_drop:
        dataset = dataset.remove_columns(to_drop)

    logger.info(
        "Custom mapping applied: input_cols=%s output_cols=%s → columns now: %s",
        input_cols, output_cols, dataset.column_names,
    )
    return dataset


def _format_dataset_for_training(dataset, tokenizer, model_name: str, config: dict):
    """
    Full pipeline mirroring unsloth-main:
    1. Apply custom column mapping if provided (renames cols to standard names)
    2. Detect format (alpaca/sharegpt/qa/completion/text/unknown)
    3. Set up tokenizer chat template for the specific model
    4. Convert every row to "text" column, appending EOS token
    5. Inject system_prompt into every example if provided
    """
    SEP = "─" * 72

    # ── Apply custom column mapping ──
    custom_mapping = config.get("custom_format_mapping") or {}
    format_type    = (config.get("format_type") or "").lower().strip()

    if custom_mapping:
        dataset = _apply_custom_mapping(dataset, custom_mapping)

    # "custom" format: user has defined column roles explicitly.
    # Skip auto-detection — treat as alpaca (instruction+output) so the
    # model's own chat template is applied with no further format conversion.
    if format_type == "custom" and custom_mapping:
        fmt = "alpaca"
        logger.info("Format override: custom → using alpaca path with user-defined column mapping")
    else:
        fmt = _detect_dataset_format(dataset)

    logger.info("Detected dataset format: %s (columns: %s)", fmt, dataset.column_names)

    # ── DIAGNOSTIC LOG 1: columns + FULL first row after mapping ──
    try:
        first_row = dict(next(iter(dataset)))
        sys_prompt_preview = (config.get("system_prompt") or "").strip()
        logger.info(
            "\n%s\n[DATASET FORMAT CHECK — after column mapping]\n"
            "  Format    : %s\n"
            "  Columns   : %s\n"
            "  Total rows: %d\n"
            "  System prompt (%d chars):\n%s",
            SEP, fmt, dataset.column_names, len(dataset),
            len(sys_prompt_preview),
            "\n".join(f"    {line}" for line in sys_prompt_preview.splitlines()) or "    (empty)",
        )
        logger.info("  --- First row (full values) ---")
        for col, val in first_row.items():
            full_val = str(val)
            logger.info("  [%s] (%d chars):\n%s",
                        col, len(full_val),
                        "\n".join(f"    {line}" for line in full_val.splitlines()))
        logger.info(SEP)
    except Exception as _e:
        logger.debug("Diagnostic log 1 failed: %s", _e)

    if fmt != "text":
        tokenizer = _setup_tokenizer_template(model_name, tokenizer)

    eos = getattr(tokenizer, "eos_token", "") or ""

    # ── text: just add EOS token ──
    if fmt == "text":
        if eos:
            def _add_eos(examples):
                return {"text": [t + eos if t and not t.endswith(eos) else (t or "")
                                 for t in examples["text"]]}
            dataset = dataset.map(_add_eos, batched=True, num_proc=_nproc())
            logger.info("Added EOS token to %d text samples", len(dataset))
        return dataset

    # ── alpaca ──
    if fmt == "alpaca":
        inst_col   = _get_original_column(dataset, "instruction")
        inp_col    = _get_original_column(dataset, "input")
        out_col    = _get_original_column(dataset, "output", "response")
        sys_prompt = (config.get("system_prompt") or "").strip()

        # ── DIAGNOSTIC LOG 2: full message structure before chat template ──
        try:
            sample_inst = str(dataset[inst_col][0] or "")
            sample_inp  = str(dataset[inp_col][0]  or "") if inp_col and inp_col in dataset.column_names else ""
            sample_out  = str(dataset[out_col][0]  or "") if out_col else ""
            sample_user = sample_inst + ("\n" + sample_inp if sample_inp.strip() else "")
            sample_msgs = []
            if sys_prompt:
                sample_msgs.append({"role": "system",    "content": sys_prompt})
            sample_msgs += [{"role": "user",      "content": sample_user},
                            {"role": "assistant", "content": sample_out}]

            parts = []
            for m in sample_msgs:
                role_header = f"  ┌─ {m['role'].upper()} ({len(m['content'])} chars) ─"
                body = "\n".join(f"  │  {line}" for line in m["content"].splitlines())
                parts.append(f"{role_header}\n{body}")

            logger.info("\n%s\n[MESSAGE STRUCTURE — before chat template (first example)]\n\n%s\n\n%s",
                        SEP, "\n\n".join(parts), SEP)
        except Exception as _e:
            logger.debug("Diagnostic log 2 failed: %s", _e)

        _logged_sample = [False]   # flag: log first formatted example only

        def _fmt_alpaca(examples):
            n = len(examples[inst_col])
            inps = examples.get(inp_col, [None] * n) if inp_col else [None] * n
            texts = []
            for inst, inp, out in zip(examples[inst_col], inps, examples[out_col]):
                user = str(inst or "")
                if inp and str(inp).strip():
                    user += "\n" + str(inp)
                msgs = []
                if sys_prompt:
                    msgs.append({"role": "system", "content": sys_prompt})
                msgs += [{"role": "user",      "content": user},
                         {"role": "assistant", "content": str(out or "")}]
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False)
                except Exception:
                    prefix = f"System: {sys_prompt}\n\n" if sys_prompt else ""
                    text = f"{prefix}### Instruction:\n{user}\n\n### Response:\n{out or ''}"
                texts.append(text + eos)

                # ── DIAGNOSTIC LOG 3: full final chat-template output ──
                if not _logged_sample[0]:
                    _logged_sample[0] = True
                    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
                    logger.info(
                        "\n%s\n[FINAL TRAINING FORMAT — %s chat template (first example)]\n\n%s\n%s",
                        SEP, model_short, texts[-1], SEP,
                    )

            return {"text": texts}

        return dataset.map(_fmt_alpaca, batched=True,
                           remove_columns=dataset.column_names, num_proc=_nproc())

    # ── sharegpt / messages ──
    if fmt == "sharegpt":
        conv_col = _get_original_column(dataset, "conversations", "messages")

        def _fmt_sharegpt(examples):
            texts = []
            for conv in examples[conv_col]:
                if not isinstance(conv, list):
                    texts.append(eos)
                    continue
                msgs = []
                for msg in conv:
                    if isinstance(msg, dict):
                        raw_role = msg.get("role", msg.get("from", "user")).lower()
                        role = _ROLE_ALIASES.get(raw_role, raw_role)
                        content = str(msg.get("content", msg.get("value", "")) or "")
                        msgs.append({"role": role, "content": content})
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False)
                except Exception:
                    text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
                texts.append(text + eos)
            return {"text": texts}

        return dataset.map(_fmt_sharegpt, batched=True,
                           remove_columns=dataset.column_names, num_proc=_nproc())

    # ── QA ──
    if fmt == "qa":
        q_col      = _get_original_column(dataset, "question", "query")
        a_col      = _get_original_column(dataset, "answer", "response")
        sys_prompt = (config.get("system_prompt") or "").strip()

        def _fmt_qa(examples):
            texts = []
            for q, a in zip(examples[q_col], examples[a_col]):
                msgs = []
                if sys_prompt:
                    msgs.append({"role": "system", "content": sys_prompt})
                msgs += [{"role": "user",      "content": str(q or "")},
                         {"role": "assistant", "content": str(a or "")}]
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False)
                except Exception:
                    text = f"Question: {q}\nAnswer: {a}"
                texts.append(text + eos)
            return {"text": texts}

        return dataset.map(_fmt_qa, batched=True,
                           remove_columns=dataset.column_names, num_proc=_nproc())

    # ── prompt/completion ──
    if fmt == "completion":
        p_col = _get_original_column(dataset, "prompt", "input")
        c_col = _get_original_column(dataset, "completion", "output")

        def _fmt_completion(examples):
            texts = []
            for p, c in zip(examples[p_col], examples[c_col]):
                msgs = [{"role": "user",      "content": str(p or "")},
                        {"role": "assistant", "content": str(c or "")}]
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False)
                except Exception:
                    text = f"{p}\n{c}"
                texts.append(text + eos)
            return {"text": texts}

        return dataset.map(_fmt_completion, batched=True,
                           remove_columns=dataset.column_names, num_proc=_nproc())

    # ── unknown: single string col or concatenate ──
    logger.warning("Unknown dataset format — attempting auto-detection")
    str_cols = [c for c in dataset.column_names
                if hasattr(dataset.features[c], "dtype")
                and dataset.features[c].dtype == "string"
                and c.lower() not in ("id", "idx", "index", "source", "split")]
    if len(str_cols) == 1:
        dataset = dataset.rename_column(str_cols[0], "text")
        if eos:
            def _add_eos2(examples):
                return {"text": [t + eos if t and not t.endswith(eos) else (t or "")
                                 for t in examples["text"]]}
            dataset = dataset.map(_add_eos2, batched=True, num_proc=_nproc())
        return dataset

    all_str = [c for c in dataset.column_names
               if hasattr(dataset.features[c], "dtype")
               and dataset.features[c].dtype == "string"]

    def _concat(examples):
        n = len(next(iter(examples.values())))
        return {"text": [
            "\n".join(f"{c}: {examples[c][i]}"
                      for c in all_str
                      if examples[c][i] and str(examples[c][i]).strip()) + eos
            for i in range(n)
        ]}

    return dataset.map(_concat, batched=True,
                       remove_columns=dataset.column_names, num_proc=_nproc())


def _apply_train_on_completions(trainer, tokenizer, model_name: str):
    """
    Apply train_on_responses_only masking — mirrors unsloth-main.
    Only the assistant response contributes to loss; instruction tokens masked.
    """
    try:
        from unsloth.chat_templates import train_on_responses_only
        lower = model_name.lower()
        if "llama-3" in lower or "llama3" in lower:
            instr = "<|start_header_id|>user<|end_header_id|>\n\n"
            resp  = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "gemma-3" in lower or "gemma3" in lower or "gemma" in lower:
            instr = "<start_of_turn>user\n"
            resp  = "<start_of_turn>model\n"
        elif "phi" in lower:
            instr = "<|user|>\n"
            resp  = "<|assistant|>\n"
        elif "mistral" in lower:
            instr = "[INST] "
            resp  = " [/INST]"
        else:
            # ChatML: Qwen, DeepSeek, SmolLM, etc.
            instr = "<|im_start|>user\n"
            resp  = "<|im_start|>assistant\n"
        trainer = train_on_responses_only(trainer,
                                          instruction_part=instr,
                                          response_part=resp)
        logger.info("train_on_responses_only applied (instr='%s...', resp='%s...')",
                    instr[:20], resp[:20])
    except Exception as e:
        logger.warning("Could not apply train_on_responses_only: %s — training on full sequence", e)
    return trainer


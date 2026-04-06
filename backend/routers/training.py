"""
Training routes — Phase 3.
Adapted from unsloth-main/studio/backend/routes/training.py
"""
import asyncio
import logging
import uuid as _uuid
from datetime import datetime
from typing import Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from auth.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/train", tags=["training"])


# ── Request/Response models ──

class TrainingStartRequest(BaseModel):
    model_name: str
    training_type: str = "LoRA/QLoRA"
    hf_token: str = ""
    load_in_4bit: bool = True
    max_seq_length: int = 2048
    hf_dataset: str = ""
    local_datasets: Optional[list[str]] = None
    local_eval_datasets: Optional[list[str]] = None
    format_type: str = ""
    subset: Optional[str] = None
    train_split: str = "train"
    eval_split: Optional[str] = None
    eval_steps: float = 0.0
    dataset_slice_start: Optional[int] = None
    dataset_slice_end: Optional[int] = None
    custom_format_mapping: Optional[dict] = None
    system_prompt: str = ""
    image_column: Optional[str] = None
    dataset_base_dir: Optional[str] = None
    is_dataset_image: bool = False
    is_dataset_audio: bool = False
    is_embedding: bool = False
    num_epochs: int = 3
    learning_rate: str = "2e-4"
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: Optional[int] = None
    warmup_ratio: Optional[float] = None
    max_steps: int = 0
    save_steps: int = 0
    output_dir: Optional[str] = None   # custom model name — used as folder name in outputs/
    save_strategy: str = "no"   # "no" | "epoch" | "steps" | "best"
    early_stopping_patience: int = 3
    weight_decay: float = 0.001
    random_seed: int = 3407
    packing: bool = False
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: Optional[str] = None
    gradient_checkpointing: str = "unsloth"
    use_rslora: bool = False
    use_loftq: bool = False
    train_on_completions: bool = False
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    enable_wandb: bool = False
    wandb_token: Optional[str] = None
    wandb_project: str = "unslothcraft-training"
    enable_tensorboard: bool = False
    tensorboard_dir: str = "runs"
    trust_remote_code: bool = False
    gpu_ids: Optional[list[int]] = None


class StopRequest(BaseModel):
    save: bool = True


# ── Routes ──

@router.post("/start")
async def start_training(request: TrainingStartRequest, _user=Depends(get_current_user)):
    from core.training import get_training_backend
    backend = get_training_backend()

    if backend.is_training_active():
        return {
            "job_id": backend.current_job_id or "",
            "status": "error",
            "message": "Training already in progress. Stop it before starting a new one.",
        }

    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"

    kwargs = request.model_dump()
    # Normalize training_type → load_in_4bit
    if kwargs["training_type"] != "LoRA/QLoRA":
        kwargs["load_in_4bit"] = False

    try:
        success = backend.start_training(job_id=job_id, **kwargs)
    except Exception as e:
        logger.error("Error starting training: %s", e, exc_info=True)
        raise HTTPException(500, f"Failed to start training: {e}")

    if not success:
        return {
            "job_id": backend.current_job_id or "",
            "status": "error",
            "message": backend._progress.error or "Failed to start training subprocess",
        }

    return {"job_id": job_id, "status": "queued", "message": "Training started in subprocess"}


@router.post("/stop")
async def stop_training(body: StopRequest = StopRequest(), _user=Depends(get_current_user)):
    from core.training import get_training_backend
    backend = get_training_backend()
    if not backend.is_training_active():
        return {"status": "idle", "message": "No training job is currently running"}
    backend.stop_training(save=body.save)
    return {"status": "stopped", "message": "Stop signal sent. Training will stop at the next safe checkpoint."}


@router.post("/reset")
async def reset_training(_user=Depends(get_current_user)):
    from core.training import get_training_backend
    backend = get_training_backend()
    if backend.is_training_active():
        if backend._cancel_requested:
            backend.force_terminate()
        else:
            raise HTTPException(409, "Training is still running. Stop it before resetting.")
    backend.reset()
    return {"status": "ok"}


@router.get("/status")
async def get_status(_user=Depends(get_current_user)):
    from core.training import get_training_backend
    backend = get_training_backend()
    job_id = getattr(backend, "current_job_id", "") or ""
    is_active = backend.is_training_active()
    progress = backend.trainer.get_training_progress()

    status_message = getattr(progress, "status_message", "Ready to train") or "Ready to train"
    error_message = getattr(progress, "error", None)
    trainer_stopped = getattr(backend, "_should_stop", False)

    if error_message:
        phase = "error"
    elif is_active:
        msg_lower = status_message.lower()
        if "loading" in msg_lower or "importing" in msg_lower:
            phase = "loading_model"
        elif any(k in msg_lower for k in ["preparing", "initializing", "configuring", "tokenizing"]):
            phase = "configuring"
        else:
            phase = "training"
    elif trainer_stopped:
        phase = "stopped"
    elif progress and getattr(progress, "is_completed", False):
        phase = "completed"
    else:
        phase = "idle"

    metric_history = None
    if backend.step_history:
        metric_history = {
            "steps": list(backend.step_history),
            "loss": list(backend.loss_history),
            "lr": list(backend.lr_history),
            "grad_norm": list(backend.grad_norm_history),
            "grad_norm_steps": list(backend.grad_norm_step_history),
            "eval_loss": list(backend.eval_loss_history),
            "eval_steps": list(backend.eval_step_history),
        }

    return {
        "job_id": job_id,
        "phase": phase,
        "is_training": is_active,
        "eval_enabled": backend.eval_enabled,
        "current_step": getattr(progress, "step", 0),
        "total_steps": getattr(progress, "total_steps", 0),
        "current_epoch": getattr(progress, "epoch", 0),
        "progress_percent": (
            round(getattr(progress, "step", 0) / max(getattr(progress, "total_steps", 1), 1) * 100, 1)
            if getattr(progress, "total_steps", 0) > 0 else 0
        ),
        "eta_seconds": getattr(progress, "eta_seconds", None),
        "loss": getattr(progress, "loss", None),
        "eval_loss": getattr(progress, "eval_loss", None),
        "learning_rate": getattr(progress, "learning_rate", None),
        "grad_norm": getattr(progress, "grad_norm", None),
        "status_message": status_message,
        "error": error_message,
        "metric_history": metric_history,
        "output_dir": getattr(backend, "_output_dir", None),
    }


@router.get("/metrics")
async def get_metrics(_user=Depends(get_current_user)):
    from core.training import get_training_backend
    backend = get_training_backend()
    return {
        "step": list(backend.step_history),
        "loss": list(backend.loss_history),
        "eval_loss": list(backend.eval_loss_history),
        "learning_rate": list(backend.lr_history),
        "grad_norm": list(backend.grad_norm_history),
        "epoch": [],
    }


@router.get("/progress")
async def stream_progress(request: Request, _user=Depends(get_current_user)):
    """SSE stream of training progress. Supports Last-Event-ID reconnect."""
    last_event_id = request.headers.get("last-event-id")
    resume_from_step: Optional[int] = None
    if last_event_id:
        try:
            resume_from_step = int(last_event_id)
        except ValueError:
            pass

    async def event_generator():
        from core.training import get_training_backend
        backend = get_training_backend()
        job_id = getattr(backend, "current_job_id", "") or ""

        def build_event(step, loss, lr, total_steps, epoch=None, grad_norm=None, eval_loss=None,
                        eta=None, elapsed=None, status_msg="") -> dict:
            total = max(total_steps or 0, 0)
            pct = round((step / total * 100.0), 1) if total > 0 and step >= 0 else 0.0
            return {
                "job_id": job_id,
                "step": step,
                "total_steps": total,
                "loss": loss,
                "learning_rate": lr,
                "progress_percent": pct,
                "epoch": epoch,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "grad_norm": grad_norm,
                "eval_loss": eval_loss,
                "status_message": status_msg,
            }

        def fmt(data: dict, event: str = "progress", eid: Optional[int] = None) -> str:
            import json
            lines = []
            if eid is not None:
                lines.append(f"id: {eid}")
            lines.append(f"event: {event}")
            lines.append(f"data: {json.dumps(data)}")
            lines.append(""); lines.append("")
            return "\n".join(lines)

        yield "retry: 3000\n\n"

        # Replay on reconnect
        if resume_from_step is not None and backend.step_history:
            gn_by_step = dict(zip(backend.grad_norm_step_history, backend.grad_norm_history))
            progress = backend.trainer.get_training_progress()
            for i, s in enumerate(backend.step_history):
                if s > resume_from_step:
                    ev = build_event(
                        s,
                        backend.loss_history[i] if i < len(backend.loss_history) else None,
                        backend.lr_history[i] if i < len(backend.lr_history) else None,
                        getattr(progress, "total_steps", s),
                        getattr(progress, "epoch", None),
                        gn_by_step.get(s),
                        status_msg=getattr(progress, "status_message", ""),
                    )
                    yield fmt(ev, "progress", s)

        # Live loop
        last_step = resume_from_step if resume_from_step is not None else -1
        no_update_count = 0

        while backend.is_training_active():
            try:
                progress = backend.trainer.get_training_progress()
                total = getattr(progress, "total_steps", 0)
                epoch = getattr(progress, "epoch", None)
                eta = getattr(progress, "eta_seconds", None)
                elapsed = getattr(progress, "elapsed_seconds", None)
                status = getattr(progress, "status_message", "")
                gn = getattr(progress, "grad_norm", None)
                el = getattr(progress, "eval_loss", None)

                if backend.step_history:
                    cur_step = backend.step_history[-1]
                    cur_loss = backend.loss_history[-1] if backend.loss_history else None
                    cur_lr = backend.lr_history[-1] if backend.lr_history else None

                    if cur_step != last_step:
                        ev = build_event(cur_step, cur_loss, cur_lr, total, epoch, gn, el, eta, elapsed, status)
                        yield fmt(ev, "progress", cur_step)
                        last_step = cur_step
                        no_update_count = 0
                    else:
                        no_update_count += 1
                        if no_update_count % 10 == 0:
                            ev = build_event(cur_step, cur_loss, cur_lr, total, epoch, gn, el, eta, elapsed, status)
                            yield fmt(ev, "heartbeat", cur_step)
                else:
                    no_update_count += 1
                    if no_update_count % 5 == 0:
                        ev = build_event(0, None, None, total, epoch, status_msg=status)
                        yield fmt(ev, "heartbeat", 0)

                if no_update_count > 1800:
                    ev = build_event(last_step, None, None, 0, status_msg="Stream timeout")
                    yield fmt(ev, "error", max(last_step, 0))
                    break

                await asyncio.sleep(1)
            except Exception as e:
                logger.error("Error in progress stream: %s", e)
                yield fmt(build_event(0, None, None, 0, status_msg=str(e)), "error", 0)
                break

        # Final event
        progress = backend.trainer.get_training_progress()
        final_step = backend.step_history[-1] if backend.step_history else last_step
        final_loss = backend.loss_history[-1] if backend.loss_history else None
        final_lr = backend.lr_history[-1] if backend.lr_history else None
        total = getattr(progress, "total_steps", final_step)
        epoch = getattr(progress, "epoch", None)
        ev = build_event(final_step, final_loss, final_lr, total, epoch,
                         status_msg=getattr(progress, "status_message", ""))
        yield fmt(ev, "complete", max(final_step or 0, 0))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

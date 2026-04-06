"""
Training backend — subprocess orchestrator.
Adapted from unsloth-main/studio/backend/core/training/training.py
Changes:
  - Replaced loggers/structlog with stdlib logging
  - Simplified prepare_gpu_selection (uses our utils/hardware)
  - Imports from our storage/utils paths
"""

import json as _json
import logging
import math
import multiprocessing as mp
import queue
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for subprocess safety
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_CTX = mp.get_context("spawn")

PLOT_WIDTH = 8
PLOT_HEIGHT = 3.5


@dataclass
class TrainingProgress:
    epoch: float = 0
    step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    is_training: bool = False
    is_completed: bool = False
    error: Optional[str] = None
    status_message: str = "Ready to train"
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
    grad_norm: Optional[float] = None
    num_tokens: Optional[int] = None
    eval_loss: Optional[float] = None


class TrainingBackend:
    """Training orchestration backend — subprocess-based."""

    FLUSH_THRESHOLD: int = 10

    def __init__(self):
        self._proc: Optional[mp.Process] = None
        self._event_queue: Any = None
        self._stop_queue: Any = None
        self._pump_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._progress = TrainingProgress()
        self._should_stop = False
        self._cancel_requested = False

        self.loss_history: list = []
        self.lr_history: list = []
        self.step_history: list = []
        self.grad_norm_history: list = []
        self.grad_norm_step_history: list = []
        self.eval_loss_history: list = []
        self.eval_step_history: list = []
        self.eval_enabled: bool = False
        self.current_theme: str = "light"

        self.current_job_id: Optional[str] = None
        self._output_dir: Optional[str] = None

        self._metric_buffer: list = []
        self._run_finalized: bool = False
        self._db_run_created: bool = False
        self._db_total_steps_set: bool = False
        self._db_config: Optional[dict] = None
        self._db_started_at: Optional[str] = None

        logger.info("TrainingBackend initialized (subprocess mode)")

    # ── Public API ──

    def start_training(self, job_id: str, **kwargs) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                logger.warning("Training subprocess already running")
                return False

        if self._pump_thread is not None and self._pump_thread.is_alive():
            self._pump_thread.join(timeout=5.0)
            if self._pump_thread.is_alive():
                logger.warning("Previous pump thread did not exit — refusing to start")
                return False
        self._pump_thread = None

        config = {
            "model_name": kwargs["model_name"],
            "training_type": kwargs.get("training_type", "LoRA/QLoRA"),
            "hf_token": kwargs.get("hf_token", ""),
            "load_in_4bit": kwargs.get("load_in_4bit", True),
            "max_seq_length": kwargs.get("max_seq_length", 2048),
            "hf_dataset": kwargs.get("hf_dataset", ""),
            "local_datasets": kwargs.get("local_datasets"),
            "local_eval_datasets": kwargs.get("local_eval_datasets"),
            "format_type": kwargs.get("format_type", ""),
            "subset": kwargs.get("subset"),
            "train_split": kwargs.get("train_split", "train"),
            "eval_split": kwargs.get("eval_split"),
            "eval_steps": kwargs.get("eval_steps", 0.0),
            "dataset_slice_start": kwargs.get("dataset_slice_start"),
            "dataset_slice_end": kwargs.get("dataset_slice_end"),
            "custom_format_mapping": kwargs.get("custom_format_mapping"),
            "system_prompt": kwargs.get("system_prompt", ""),
            "image_column": kwargs.get("image_column") or "",
            "dataset_base_dir": kwargs.get("dataset_base_dir") or "",
            "is_dataset_image": kwargs.get("is_dataset_image", False),
            "is_dataset_audio": kwargs.get("is_dataset_audio", False),
            "is_embedding": kwargs.get("is_embedding", False),
            "num_epochs": kwargs.get("num_epochs", 3),
            "learning_rate": kwargs.get("learning_rate", "2e-4"),
            "batch_size": kwargs.get("batch_size", 2),
            "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 4),
            "warmup_steps": kwargs.get("warmup_steps"),
            "warmup_ratio": kwargs.get("warmup_ratio"),
            "max_steps": kwargs.get("max_steps", 0),
            "save_steps": kwargs.get("save_steps", 0),
            "output_dir": kwargs.get("output_dir") or None,
            "save_strategy": kwargs.get("save_strategy", "no"),
            "early_stopping_patience": kwargs.get("early_stopping_patience", 3),
            "weight_decay": kwargs.get("weight_decay", 0.001),
            "random_seed": kwargs.get("random_seed", 3407),
            "packing": kwargs.get("packing", False),
            "optim": kwargs.get("optim", "adamw_8bit"),
            "lr_scheduler_type": kwargs.get("lr_scheduler_type", "linear"),
            "use_lora": kwargs.get("use_lora", True),
            "lora_r": kwargs.get("lora_r", 16),
            "lora_alpha": kwargs.get("lora_alpha", 16),
            "lora_dropout": kwargs.get("lora_dropout", 0.0),
            "target_modules": kwargs.get("target_modules"),
            "gradient_checkpointing": kwargs.get("gradient_checkpointing", "unsloth"),
            "use_rslora": kwargs.get("use_rslora", False),
            "use_loftq": kwargs.get("use_loftq", False),
            "train_on_completions": kwargs.get("train_on_completions", False),
            "finetune_vision_layers": kwargs.get("finetune_vision_layers", True),
            "finetune_language_layers": kwargs.get("finetune_language_layers", True),
            "finetune_attention_modules": kwargs.get("finetune_attention_modules", True),
            "finetune_mlp_modules": kwargs.get("finetune_mlp_modules", True),
            "enable_wandb": kwargs.get("enable_wandb", False),
            "wandb_token": kwargs.get("wandb_token"),
            "wandb_project": kwargs.get("wandb_project", "unslothcraft-training"),
            "enable_tensorboard": kwargs.get("enable_tensorboard", False),
            "tensorboard_dir": kwargs.get("tensorboard_dir", "runs"),
            "trust_remote_code": kwargs.get("trust_remote_code", False),
            "gpu_ids": kwargs.get("gpu_ids"),
        }

        if config["training_type"] != "LoRA/QLoRA":
            config["load_in_4bit"] = False

        # GPU selection
        from utils.hardware import prepare_gpu_selection
        resolved_gpu_ids, gpu_selection = prepare_gpu_selection(
            kwargs.get("gpu_ids"),
            model_name=config["model_name"],
        )
        config["resolved_gpu_ids"] = resolved_gpu_ids
        config["gpu_selection"] = gpu_selection

        from core.training.worker import run_training_process

        event_queue = _CTX.Queue()
        stop_queue = _CTX.Queue()

        proc = _CTX.Process(
            target=run_training_process,
            kwargs={"event_queue": event_queue, "stop_queue": stop_queue, "config": config},
            daemon=True,
        )
        try:
            proc.start()
        except Exception:
            logger.error("Failed to start training subprocess", exc_info=True)
            return False

        logger.info("Training subprocess started (pid=%s)", proc.pid)

        self.current_job_id = job_id
        self._should_stop = False
        self._cancel_requested = False
        self._progress = TrainingProgress(is_training=True, status_message="Initializing training...")
        self.loss_history.clear()
        self.lr_history.clear()
        self.step_history.clear()
        self.grad_norm_history.clear()
        self.grad_norm_step_history.clear()
        self.eval_loss_history.clear()
        self.eval_step_history.clear()
        self.eval_enabled = False
        self._output_dir = None
        self._metric_buffer.clear()
        self._run_finalized = False
        self._db_run_created = False
        self._db_total_steps_set = False
        self._db_config = {k: v for k, v in config.items() if k not in {"hf_token", "wandb_token"}}
        self._db_started_at = datetime.now(timezone.utc).isoformat()

        self._event_queue = event_queue
        self._stop_queue = stop_queue
        self._proc = proc

        self._ensure_db_run_created()

        self._pump_thread = threading.Thread(target=self._pump_loop, daemon=True)
        self._pump_thread.start()

        return True

    def stop_training(self, save: bool = True) -> bool:
        self._should_stop = True
        if not save:
            self._cancel_requested = True
        with self._lock:
            if self._stop_queue is not None:
                try:
                    self._stop_queue.put({"type": "stop", "save": save})
                except (OSError, ValueError):
                    pass
            self._progress.status_message = (
                "Stopping training and saving checkpoint..." if save else "Cancelling training..."
            )
        return True

    def reset(self) -> None:
        """Clear all training state so a new run can start cleanly."""
        self._should_stop = False
        self._cancel_requested = False
        self._output_dir = None
        self._progress = TrainingProgress()
        self.loss_history.clear()
        self.lr_history.clear()
        self.step_history.clear()
        self.grad_norm_history.clear()
        self.grad_norm_step_history.clear()
        self.eval_loss_history.clear()
        self.eval_step_history.clear()
        self._metric_buffer.clear()
        self._run_finalized = False
        self._db_run_created = False
        self._db_total_steps_set = False

    def force_terminate(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                self._proc.terminate()
            proc = self._proc
        if proc is not None:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2.0)
        if self._pump_thread is not None and self._pump_thread.is_alive():
            self._pump_thread.join(timeout=8.0)

    def is_training_active(self) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                return True
            if self._should_stop:
                return False
            p = self._progress
            if p.is_training:
                return True
            if p.is_completed or p.error:
                return False
            status_lower = (p.status_message or "").lower()
            if any(k in status_lower for k in ["cancelled", "canceled", "stopped", "completed", "ready to train"]):
                return False
            if any(k in status_lower for k in ["loading", "preparing", "training", "configuring", "tokenizing", "starting", "importing"]):
                return True
            return False

    def get_training_status(self, theme: str = "light") -> Tuple:
        with self._lock:
            progress = self._progress
        if not (progress.is_training or progress.is_completed or progress.error):
            return (None, progress)
        plot = self._create_loss_plot(progress, theme)
        return (plot, progress)

    # ── Compatibility shim ──

    class _TrainerShim:
        def __init__(self, backend: "TrainingBackend"):
            self._backend = backend
            self.should_stop = False

        @property
        def training_progress(self):
            return self._backend._progress

        @training_progress.setter
        def training_progress(self, value):
            self._backend._progress = value

        def get_training_progress(self):
            return self._backend._progress

        def _update_progress(self, **kwargs):
            with self._backend._lock:
                for key, value in kwargs.items():
                    if hasattr(self._backend._progress, key):
                        setattr(self._backend._progress, key, value)

    @property
    def trainer(self):
        return self._TrainerShim(self)

    # ── Event pump ──

    def _pump_loop(self) -> None:
        while True:
            if self._proc is None or self._event_queue is None:
                return
            event = self._read_queue(self._event_queue, timeout_sec=0.25)
            if event is not None:
                self._handle_event(event)
                continue
            if self._proc.is_alive():
                continue
            for e in self._drain_queue(self._event_queue):
                self._handle_event(e)
            with self._lock:
                if self._progress.is_training:
                    if self._should_stop:
                        self._progress.is_training = False
                        self._progress.status_message = "Training stopped."
                    else:
                        self._progress.is_training = False
                        self._progress.error = self._progress.error or "Training process exited unexpectedly"
            self._ensure_db_run_created()
            self._finalize_run_in_db(
                status="stopped" if self._should_stop else "error",
                error_message=None if self._should_stop else "Training process terminated unexpectedly",
            )
            return

    def _handle_event(self, event: dict) -> None:
        etype = event.get("type")
        db_action: Optional[str] = None
        db_action_kwargs: dict = {}

        with self._lock:
            if etype == "progress":
                self._progress.step = event.get("step", self._progress.step)
                self._progress.epoch = event.get("epoch", self._progress.epoch)
                _raw_loss = event.get("loss")
                _raw_lr = event.get("learning_rate")
                try:
                    _safe_loss = float(_raw_loss) if _raw_loss is not None else None
                except (TypeError, ValueError):
                    _safe_loss = None
                if _safe_loss is not None and not math.isfinite(_safe_loss):
                    _safe_loss = None
                try:
                    _safe_lr = float(_raw_lr) if _raw_lr is not None else None
                except (TypeError, ValueError):
                    _safe_lr = None
                if _safe_lr is not None and not math.isfinite(_safe_lr):
                    _safe_lr = None
                if _safe_loss is not None:
                    self._progress.loss = _safe_loss
                if _safe_lr is not None:
                    self._progress.learning_rate = _safe_lr
                self._progress.total_steps = event.get("total_steps", self._progress.total_steps)
                self._progress.elapsed_seconds = event.get("elapsed_seconds")
                self._progress.eta_seconds = event.get("eta_seconds")
                self._progress.grad_norm = event.get("grad_norm")
                self._progress.num_tokens = event.get("num_tokens")
                self._progress.eval_loss = event.get("eval_loss")
                self._progress.is_training = True
                status = event.get("status_message", "")
                if status:
                    self._progress.status_message = status

                step = event.get("step", 0)
                loss = _safe_loss
                lr = _safe_lr
                if step > 0 and loss is not None:
                    self.loss_history.append(loss)
                    self.lr_history.append(lr if lr is not None else 0.0)
                    self.step_history.append(step)

                grad_norm = event.get("grad_norm")
                gn = None
                if grad_norm is not None:
                    try:
                        gn = float(grad_norm)
                    except (TypeError, ValueError):
                        gn = None
                    if step > 0 and gn is not None and math.isfinite(gn):
                        self.grad_norm_history.append(gn)
                        self.grad_norm_step_history.append(step)
                    else:
                        gn = None

                eval_loss_val = event.get("eval_loss")
                if eval_loss_val is not None:
                    try:
                        eval_loss_val = float(eval_loss_val)
                    except (TypeError, ValueError):
                        eval_loss_val = None
                    if step > 0 and eval_loss_val is not None and math.isfinite(eval_loss_val):
                        self.eval_loss_history.append(eval_loss_val)
                        self.eval_step_history.append(step)
                        self.eval_enabled = True
                    else:
                        eval_loss_val = None

                self._metric_buffer.append({
                    "step": step, "loss": loss, "learning_rate": lr,
                    "grad_norm": gn, "eval_loss": eval_loss_val,
                    "epoch": event.get("epoch"), "num_tokens": event.get("num_tokens"),
                    "elapsed_seconds": event.get("elapsed_seconds"),
                })

                if not self._db_run_created and self.current_job_id and self._db_config:
                    db_action = "create_run"
                    db_action_kwargs = {
                        "job_id": self.current_job_id,
                        "model_name": self._db_config["model_name"],
                        "dataset_name": self._db_config.get("hf_dataset") or next(iter(self._db_config.get("local_datasets") or []), "unknown"),
                        "config_json": _json.dumps(self._db_config),
                        "started_at": self._db_started_at or datetime.now(timezone.utc).isoformat(),
                        "total_steps": event.get("total_steps"),
                    }
                elif event.get("total_steps") and self._db_run_created and not self._db_total_steps_set:
                    db_action = "update_total_steps"
                    db_action_kwargs = {"job_id": self.current_job_id, "total_steps": event["total_steps"]}
                elif len(self._metric_buffer) >= self.FLUSH_THRESHOLD:
                    db_action = "flush"

            elif etype == "eval_configured":
                self.eval_enabled = True
            elif etype == "status":
                self._progress.status_message = event.get("message", "")
                self._progress.is_training = True
            elif etype == "complete":
                self._progress.is_training = False
                self._progress.is_completed = True
                self._output_dir = event.get("output_dir")
                self._progress.status_message = event.get("status_message", "Training completed")
                db_action = "create_and_finalize" if not self._db_run_created else "finalize"
                db_action_kwargs = {"status": "stopped" if self._should_stop else "completed", "output_dir": self._output_dir}
            elif etype == "error":
                self._progress.is_training = False
                self._progress.error = event.get("error", "Unknown error")
                logger.error("Training error: %s", event.get("error"))
                db_action = "create_and_finalize" if not self._db_run_created else "finalize"
                db_action_kwargs = {"status": "stopped" if self._should_stop else "error", "error_message": event.get("error")}

        # DB I/O outside lock
        if db_action == "create_run":
            try:
                from storage.studio_db import create_run
                create_run(id=db_action_kwargs["job_id"], model_name=db_action_kwargs["model_name"],
                           dataset_name=db_action_kwargs["dataset_name"], config_json=db_action_kwargs["config_json"],
                           started_at=db_action_kwargs["started_at"], total_steps=db_action_kwargs["total_steps"])
                self._db_run_created = True
                if db_action_kwargs["total_steps"]:
                    self._db_total_steps_set = True
            except Exception:
                logger.warning("Failed to create DB run record", exc_info=True)
        elif db_action == "create_and_finalize":
            self._ensure_db_run_created()
            self._finalize_run_in_db(**db_action_kwargs)
        elif db_action == "update_total_steps":
            try:
                from storage.studio_db import update_run_total_steps
                update_run_total_steps(db_action_kwargs["job_id"], db_action_kwargs["total_steps"])
                self._db_total_steps_set = True
            except Exception:
                logger.warning("Failed to update total_steps", exc_info=True)
        elif db_action == "flush":
            self._flush_metrics_to_db()
        elif db_action == "finalize":
            self._finalize_run_in_db(**db_action_kwargs)

    def _ensure_db_run_created(self) -> None:
        if self._db_run_created or not self.current_job_id or not self._db_config:
            return
        try:
            from storage.studio_db import create_run
            dataset_name = self._db_config.get("hf_dataset") or next(iter(self._db_config.get("local_datasets") or []), "unknown")
            create_run(id=self.current_job_id, model_name=self._db_config["model_name"],
                       dataset_name=dataset_name, config_json=_json.dumps(self._db_config),
                       started_at=self._db_started_at or datetime.now(timezone.utc).isoformat(),
                       total_steps=self._progress.total_steps or None)
            self._db_run_created = True
        except Exception:
            logger.warning("Failed to create DB run record for early failure", exc_info=True)

    def _finalize_run_in_db(self, status: str, error_message: Optional[str] = None, output_dir: Optional[str] = None) -> None:
        if not self.current_job_id or not self._db_run_created or self._run_finalized:
            return
        self._flush_metrics_to_db()
        try:
            from storage.studio_db import finish_run
            from utils.downsample import downsample
            sparkline = downsample(self.loss_history, 50)
            finish_run(id=self.current_job_id, status=status,
                       ended_at=datetime.now(timezone.utc).isoformat(),
                       final_step=self._progress.step,
                       final_loss=self._progress.loss if self._progress.loss is not None and math.isfinite(self._progress.loss) else None,
                       duration_seconds=self._progress.elapsed_seconds,
                       loss_sparkline=_json.dumps(sparkline),
                       output_dir=output_dir, error_message=error_message)
            self._run_finalized = True
        except Exception:
            logger.warning("Failed to finalize run in DB (status=%s)", status, exc_info=True)

    def _flush_metrics_to_db(self) -> None:
        if not self._metric_buffer or not self.current_job_id or not self._db_run_created:
            return
        if len(self._metric_buffer) > 500:
            self._metric_buffer = self._metric_buffer[-500:]
        batch = list(self._metric_buffer)
        try:
            from storage.studio_db import insert_metrics_batch, update_run_progress
            insert_metrics_batch(self.current_job_id, batch)
            del self._metric_buffer[:len(batch)]
            update_run_progress(id=self.current_job_id, step=self._progress.step,
                                loss=self._progress.loss if self._progress.loss is not None and math.isfinite(self._progress.loss) else None,
                                duration_seconds=self._progress.elapsed_seconds)
        except Exception:
            logger.warning("Failed to flush metrics to DB", exc_info=True)

    @staticmethod
    def _read_queue(q: Any, timeout_sec: float) -> Optional[dict]:
        try:
            return q.get(timeout=timeout_sec)
        except queue.Empty:
            return None
        except (EOFError, OSError, ValueError):
            return None

    @staticmethod
    def _drain_queue(q: Any) -> list:
        events = []
        while True:
            try:
                events.append(q.get_nowait())
            except queue.Empty:
                return events
            except (EOFError, OSError, ValueError):
                return events

    def _create_loss_plot(self, progress: TrainingProgress, theme: str = "light") -> plt.Figure:
        plt.close("all")
        DARK = {"facecolor": "#1E293B", "grid_color": "#334155", "line": "#00A5D9",
                "text": "#E2E8F0", "empty_text": "#94A3B8"}
        LIGHT = {"facecolor": "#ffffff", "grid_color": "#d1d5db", "line": "#0070AD",
                 "text": "#1f2937", "empty_text": "#6b7280"}
        style = DARK if theme == "dark" else LIGHT
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        fig.patch.set_facecolor(style["facecolor"])
        ax.set_facecolor(style["facecolor"])
        if self.loss_history:
            ax.scatter(self.step_history, self.loss_history, s=16, alpha=0.5, color="#60a5fa", linewidths=0)
            if len(self.loss_history) >= 2:
                window = min(20, len(self.loss_history))
                cumsum = [0.0]
                for v in self.loss_history:
                    cumsum.append(cumsum[-1] + float(v))
                ma = [(cumsum[i+1] - cumsum[max(0, i-window+1)]) / min(i+1, window) for i in range(len(self.loss_history))]
                ax.plot(self.step_history, ma, color=style["line"], linewidth=2.5, alpha=0.95, label=f"Avg ({ma[-1]:.4f})")
                leg = ax.legend(frameon=False, fontsize=9)
                for t in leg.get_texts():
                    t.set_color(style["text"])
            ax.set_xlabel("Steps", fontsize=10, color=style["text"])
            ax.set_ylabel("Loss", fontsize=10, color=style["text"])
            title = progress.status_message or f"Step {progress.step}/{progress.total_steps}"
            ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color=style["text"])
            ax.grid(True, alpha=0.4, linestyle="--", color=style["grid_color"])
            ax.tick_params(colors=style["text"], which="both")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        else:
            ax.text(0.5, 0.5, progress.status_message or "Waiting for training data...",
                    ha="center", va="center", fontsize=14, color=style["empty_text"], transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
        fig.tight_layout()
        return fig


# ── Global singleton ──
_training_backend: Optional[TrainingBackend] = None


def get_training_backend() -> TrainingBackend:
    global _training_backend
    if _training_backend is None:
        _training_backend = TrainingBackend()
    return _training_backend

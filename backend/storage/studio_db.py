"""
SQLite storage for training run history and metrics.
Adapted from unsloth-main/studio/backend/storage/studio_db.py
Changes: replaced utils.paths import with direct path resolution.
"""
import json
import logging
import os
import platform
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def _studio_db_path() -> Path:
    from utils.paths import studio_db_path
    return studio_db_path()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _denied_path_prefixes() -> list[str]:
    system = platform.system()
    if system == "Linux":
        return ["/proc", "/sys", "/dev", "/etc", "/boot", "/run"]
    if system == "Darwin":
        return ["/System", "/Library", "/dev", "/etc", "/private/etc",
                "/tmp", "/private/tmp", "/var", "/private/var"]
    if system == "Windows":
        win = os.environ.get("SystemRoot", r"C:\Windows")
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        return [os.path.normcase(p) for p in [win, pf, pf86]]
    return []


_schema_lock = threading.Lock()
_schema_ready = False


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id TEXT NOT NULL PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'running',
            model_name TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            total_steps INTEGER,
            final_step INTEGER,
            final_loss REAL,
            output_dir TEXT,
            error_message TEXT,
            duration_seconds REAL,
            loss_sparkline TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
            step INTEGER NOT NULL,
            loss REAL,
            learning_rate REAL,
            grad_norm REAL,
            eval_loss REAL,
            epoch REAL,
            num_tokens INTEGER,
            elapsed_seconds REAL,
            UNIQUE(run_id, step)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON training_metrics(run_id)")
    collation = "COLLATE NOCASE" if platform.system() == "Windows" else ""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS scan_folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE {collation},
            created_at TEXT NOT NULL
        )
    """)


def get_connection() -> sqlite3.Connection:
    global _schema_ready
    db_path = _studio_db_path()
    _ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    if not _schema_ready:
        with _schema_lock:
            if not _schema_ready:
                try:
                    _ensure_schema(conn)
                    _schema_ready = True
                except Exception:
                    conn.close()
                    raise
    return conn


# ── Training run CRUD ──

def create_run(id: str, model_name: str, dataset_name: str, config_json: str,
               started_at: str, total_steps: Optional[int]) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO training_runs (id, model_name, dataset_name, config_json, started_at, total_steps) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (id, model_name, dataset_name, config_json, started_at, total_steps),
        )
        conn.commit()
    finally:
        conn.close()


def update_run_total_steps(id: str, total_steps: int) -> None:
    conn = get_connection()
    try:
        conn.execute("UPDATE training_runs SET total_steps = ? WHERE id = ?", (total_steps, id))
        conn.commit()
    finally:
        conn.close()


def update_run_progress(id: str, step: int, loss: Optional[float], duration_seconds: Optional[float]) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE training_runs SET final_step = ?, final_loss = ?, duration_seconds = ? WHERE id = ?",
            (step, loss, duration_seconds, id),
        )
        conn.commit()
    finally:
        conn.close()


def finish_run(id: str, status: str, ended_at: str, final_step: Optional[int],
               final_loss: Optional[float], duration_seconds: Optional[float],
               loss_sparkline: Optional[str] = None, output_dir: Optional[str] = None,
               error_message: Optional[str] = None) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE training_runs SET status=?, ended_at=?, final_step=?, final_loss=?, "
            "duration_seconds=?, loss_sparkline=?, output_dir=?, error_message=? WHERE id=?",
            (status, ended_at, final_step, final_loss, duration_seconds,
             loss_sparkline, output_dir, error_message, id),
        )
        conn.commit()
    finally:
        conn.close()


def insert_metrics_batch(run_id: str, metrics: list[dict]) -> None:
    if not metrics:
        return
    conn = get_connection()
    try:
        conn.executemany(
            "INSERT INTO training_metrics "
            "(run_id, step, loss, learning_rate, grad_norm, eval_loss, epoch, num_tokens, elapsed_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id, step) DO UPDATE SET "
            "loss=COALESCE(excluded.loss, loss), "
            "learning_rate=COALESCE(excluded.learning_rate, learning_rate), "
            "grad_norm=COALESCE(excluded.grad_norm, grad_norm), "
            "eval_loss=COALESCE(excluded.eval_loss, eval_loss), "
            "epoch=COALESCE(excluded.epoch, epoch), "
            "num_tokens=COALESCE(excluded.num_tokens, num_tokens), "
            "elapsed_seconds=COALESCE(excluded.elapsed_seconds, elapsed_seconds)",
            [(run_id, m.get("step"), m.get("loss"), m.get("learning_rate"),
              m.get("grad_norm"), m.get("eval_loss"), m.get("epoch"),
              m.get("num_tokens"), m.get("elapsed_seconds")) for m in metrics],
        )
        conn.commit()
    finally:
        conn.close()


def list_runs(limit: int = 50, offset: int = 0) -> dict:
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
        rows = conn.execute(
            "SELECT id, status, model_name, dataset_name, config_json, started_at, ended_at, "
            "total_steps, final_step, final_loss, output_dir, duration_seconds, "
            "error_message, loss_sparkline FROM training_runs ORDER BY started_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        runs = []
        for row in rows:
            run = dict(row)
            if run.get("loss_sparkline"):
                try:
                    run["loss_sparkline"] = json.loads(run["loss_sparkline"])
                except Exception:
                    run["loss_sparkline"] = None
            runs.append(run)
        return {"runs": runs, "total": total}
    finally:
        conn.close()


def get_run(id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM training_runs WHERE id = ?", (id,)).fetchone()
        if row is None:
            return None
        run = dict(row)
        if run.get("loss_sparkline"):
            try:
                run["loss_sparkline"] = json.loads(run["loss_sparkline"])
            except Exception:
                run["loss_sparkline"] = None
        return run
    finally:
        conn.close()


def get_run_metrics(id: str) -> dict:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT step, loss, learning_rate, grad_norm, eval_loss, epoch, "
            "num_tokens, elapsed_seconds FROM training_metrics WHERE run_id = ? ORDER BY step",
            (id,),
        ).fetchall()
        step_h, loss_h, loss_step_h = [], [], []
        lr_h, lr_step_h = [], []
        gn_h, gn_step_h = [], []
        eval_h, eval_step_h = [], []
        final_epoch = None
        final_num_tokens = None
        for row in rows:
            s = row["step"]
            step_h.append(s)
            if s > 0 and row["loss"] is not None:
                loss_h.append(row["loss"]); loss_step_h.append(s)
            if s > 0 and row["learning_rate"] is not None:
                lr_h.append(row["learning_rate"]); lr_step_h.append(s)
            if s > 0 and row["grad_norm"] is not None:
                gn_h.append(row["grad_norm"]); gn_step_h.append(s)
            if s > 0 and row["eval_loss"] is not None:
                eval_h.append(row["eval_loss"]); eval_step_h.append(s)
            if row["epoch"] is not None:
                final_epoch = row["epoch"]
            if row["num_tokens"] is not None:
                final_num_tokens = row["num_tokens"]
        return {
            "step_history": step_h, "loss_history": loss_h, "loss_step_history": loss_step_h,
            "lr_history": lr_h, "lr_step_history": lr_step_h,
            "grad_norm_history": gn_h, "grad_norm_step_history": gn_step_h,
            "eval_loss_history": eval_h, "eval_step_history": eval_step_h,
            "final_epoch": final_epoch, "final_num_tokens": final_num_tokens,
        }
    finally:
        conn.close()


def delete_run(id: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM training_runs WHERE id = ?", (id,))
        conn.commit()
    finally:
        conn.close()


def cleanup_orphaned_runs() -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE training_runs SET status='error', error_message='Server restarted during training', "
            "ended_at=? WHERE status='running'",
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()
    finally:
        conn.close()


# ── Scan folders ──

def list_scan_folders() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT id, path, created_at FROM scan_folders ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def add_scan_folder(path: str) -> dict:
    if not path or not path.strip():
        raise ValueError("Path cannot be empty")
    normalized = os.path.realpath(os.path.expanduser(path.strip()))
    if not os.path.exists(normalized):
        raise ValueError("Path does not exist")
    if not os.path.isdir(normalized):
        raise ValueError("Path must be a directory")
    if not os.access(normalized, os.R_OK | os.X_OK):
        raise ValueError("Path is not readable")
    is_win = platform.system() == "Windows"
    check = os.path.normcase(normalized) if is_win else normalized
    for prefix in _denied_path_prefixes():
        if check == prefix or check.startswith(prefix + os.sep):
            raise ValueError(f"Path under {prefix} is not allowed")
    conn = get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        existing = conn.execute(
            "SELECT id, path, created_at FROM scan_folders WHERE path = ?" +
            (" COLLATE NOCASE" if is_win else ""),
            (normalized,),
        ).fetchone()
        if existing:
            return dict(existing)
        try:
            conn.execute("INSERT INTO scan_folders (path, created_at) VALUES (?, ?)", (normalized, now))
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        row = conn.execute(
            "SELECT id, path, created_at FROM scan_folders WHERE path = ?" +
            (" COLLATE NOCASE" if is_win else ""),
            (normalized,),
        ).fetchone()
        if row is None:
            raise ValueError("Folder was concurrently removed")
        return dict(row)
    finally:
        conn.close()


def remove_scan_folder(id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM scan_folders WHERE id = ?", (id,))
        conn.commit()
    finally:
        conn.close()

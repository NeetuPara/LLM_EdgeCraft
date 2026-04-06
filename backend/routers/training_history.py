"""Training history endpoints — Phase 3."""
import logging
from fastapi import APIRouter, Depends, HTTPException
from auth.dependencies import get_current_user
from storage.studio_db import list_runs, get_run, get_run_metrics, delete_run

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/train", tags=["training-history"])


@router.get("/runs")
def list_training_runs(limit: int = 50, offset: int = 0, _user=Depends(get_current_user)):
    """List all training runs, newest first."""
    return list_runs(limit=limit, offset=offset)


@router.get("/runs/{run_id}")
def get_training_run(run_id: str, _user=Depends(get_current_user)):
    """Get a single training run with its metric history."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Training run '{run_id}' not found")
    metrics = get_run_metrics(run_id)
    return {**run, "metrics": metrics}


@router.delete("/runs/{run_id}")
def delete_training_run(run_id: str, _user=Depends(get_current_user)):
    """Delete a training run (refuses if status is 'running')."""
    from core.training import get_training_backend
    backend = get_training_backend()
    if backend.current_job_id == run_id and backend.is_training_active():
        raise HTTPException(409, "Cannot delete a currently running training job. Stop it first.")
    run = get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Training run '{run_id}' not found")
    delete_run(run_id)
    return {"msg": f"Deleted run {run_id}"}

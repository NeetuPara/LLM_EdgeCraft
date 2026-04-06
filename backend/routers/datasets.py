"""
Datasets routes — Phase 6.
Adapted from unsloth-main/studio/backend/routes/datasets.py
"""
import logging
import json
import shutil
import zipfile
from itertools import islice
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from auth.dependencies import get_current_user
from utils.datasets import check_dataset_format, serialize_preview_rows

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# Supported upload extensions
_UPLOAD_EXTS = {".csv", ".json", ".jsonl", ".parquet", ".zip"}
_TABULAR_EXTS = (".parquet", ".json", ".jsonl", ".csv", ".tsv", ".arrow")
_ARCHIVE_EXTS = (".tar", ".tar.gz", ".tgz", ".gz", ".zst", ".zip", ".txt")
DATA_EXTS = _TABULAR_EXTS + _ARCHIVE_EXTS

PREVIEW_SIZE = 10


# ── Request models ──

class CheckFormatRequest(BaseModel):
    dataset_name: str
    train_split: str = "train"
    subset: Optional[str] = None
    hf_token: Optional[str] = None
    is_vlm: bool = False


class AiAssistRequest(BaseModel):
    dataset_name: str
    columns: list[str]
    samples: list[dict]
    hf_token: Optional[str] = None
    model_name: Optional[str] = None
    model_type: Optional[str] = None


# ── Helpers ──

def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    return name or "dataset_upload"


def _extract_zip(zip_bytes: bytes, stem: str, upload_dir: Path) -> Path:
    """
    Extract a zip archive into upload_dir / stem /.
    Guards against path traversal. Returns the folder path.
    """
    folder = upload_dir / stem
    folder.mkdir(parents=True, exist_ok=True)
    folder_resolved = folder.resolve()

    with zipfile.ZipFile(__import__("io").BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            # Guard against path traversal
            member_path = (folder / member.filename).resolve()
            if not str(member_path).startswith(str(folder_resolved)):
                logger.warning("Skipping unsafe zip entry: %s", member.filename)
                continue
            zf.extract(member, folder)

    logger.info("Extracted zip '%s' → %s (%d entries)", stem, folder, len(zf.infolist()))
    return folder


def _serialize_preview_value(value):
    """Make a value JSON-safe for client preview."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _serialize_preview_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_preview_value(v) for v in value]
    return str(value)


def _serialize_rows(rows) -> list[dict]:
    return [
        {str(k): _serialize_preview_value(v) for k, v in dict(row).items()}
        for row in rows
    ]


def _load_local_preview(dataset_path: Path, split: str, n: int):
    """Load preview rows from a local file or folder (including zip-extracted folders)."""
    from datasets import load_dataset

    if dataset_path.is_dir():
        # 1. Prefer dedicated parquet-files/ subfolder (recipe datasets)
        parquet_dir = dataset_path / "parquet-files" if (dataset_path / "parquet-files").exists() else None

        # 2. Collect all parquet files — rglob covers nested zip structures
        if parquet_dir:
            parquet_files = sorted(parquet_dir.rglob("*.parquet"))
        else:
            parquet_files = sorted(dataset_path.rglob("*.parquet"))

        if parquet_files:
            ds = load_dataset("parquet", data_files=[str(p) for p in parquet_files], split=split)
            total = len(ds)
            return ds.select(range(min(n, total))), total

        # 3. Find any supported tabular file (JSON / JSONL / CSV) — rglob for nested zips
        for ext in (".json", ".jsonl", ".csv"):
            files = sorted(dataset_path.rglob(f"*{ext}"))
            if files:
                dataset_path = files[0]
                break
        else:
            raise HTTPException(400, f"No supported data files found in folder: {dataset_path}")

    ext = dataset_path.suffix.lower()
    loader = {".json": "json", ".jsonl": "json", ".csv": "csv", ".tsv": "csv",
              ".parquet": "parquet", ".arrow": "arrow"}.get(ext, "json")
    ds = load_dataset(loader, data_files=str(dataset_path), split=split)
    total = len(ds)
    return ds.select(range(min(n, total))), total


def _load_hf_preview(dataset_name: str, split: str, subset: Optional[str],
                     hf_token: Optional[str], n: int):
    """Load preview rows from a HuggingFace dataset (streaming)."""
    from datasets import Dataset, load_dataset

    # Tier 1: list_repo_files → load only the first data file (fast, ~2-4s)
    preview_slice = None
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_files = list(api.list_repo_files(dataset_name, repo_type="dataset", token=hf_token or None))
        data_files = [f for f in repo_files if any(f.endswith(ext) for ext in DATA_EXTS)]
        tabular = [f for f in data_files if any(f.endswith(ext) for ext in _TABULAR_EXTS)]
        candidates = tabular or data_files
        if subset and candidates:
            matches = [f for f in candidates if subset in Path(f).stem]
            if matches:
                candidates = matches
        if candidates:
            kwargs = {
                "path": dataset_name, "data_files": [candidates[0]],
                "split": "train", "streaming": True,
            }
            if hf_token:
                kwargs["token"] = hf_token
            rows = list(islice(load_dataset(**kwargs), n))
            if rows:
                preview_slice = Dataset.from_list(rows)
    except Exception as e:
        logger.debug("Tier 1 failed: %s", e)

    if preview_slice is None:
        # Tier 2: full streaming fallback
        kwargs = {"path": dataset_name, "split": split, "streaming": True}
        if subset:
            kwargs["name"] = subset
        if hf_token:
            kwargs["token"] = hf_token
        rows = list(islice(load_dataset(**kwargs), n))
        if not rows:
            raise HTTPException(400, "Dataset appears empty or could not be streamed")
        preview_slice = Dataset.from_list(rows)

    return preview_slice, None  # total unknown for HF streaming


# ── Routes ──

@router.post("/check-format")
def check_format(request: CheckFormatRequest, _user=Depends(get_current_user)):
    """
    Inspect a dataset (HF or local), detect its format, return preview rows.
    Runs synchronously in FastAPI's thread pool (no async).
    """
    try:
        from utils.paths import dataset_uploads_root, recipe_datasets_root

        # Check if local path
        dataset_path = Path(request.dataset_name)
        uploads = dataset_uploads_root()
        recipes = recipe_datasets_root()
        local_candidates = [
            dataset_path,
            uploads / request.dataset_name,
            recipes / request.dataset_name,
        ]
        local_path = next((p for p in local_candidates if p.exists()), None)

        total_rows = None
        if local_path:
            preview_slice, total_rows = _load_local_preview(
                local_path, request.train_split or "train", PREVIEW_SIZE
            )
        else:
            preview_slice, total_rows = _load_hf_preview(
                request.dataset_name, request.train_split or "train",
                request.subset, request.hf_token, PREVIEW_SIZE,
            )

        # Format detection
        result = check_dataset_format(preview_slice, is_vlm=request.is_vlm)

        # Ensure preview samples populated
        if not result.get("preview_samples"):
            result["preview_samples"] = _serialize_rows(preview_slice)

        result["total_rows"] = total_rows
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error checking dataset format: %s", e, exc_info=True)
        raise HTTPException(500, f"Failed to check dataset format: {e}")


@router.post("/upload")
async def upload_dataset(
    file: UploadFile,
    action: str = Query(default="ask"),   # "ask" | "replace" | "new"
    _user=Depends(get_current_user),
):
    """Upload a local dataset file (CSV/JSON/JSONL/Parquet) or a zip of multiple files.

    action="ask"     → if file/folder exists, return {conflict:true} without saving
    action="replace" → overwrite existing file / re-extract zip (deletes old folder)
    action="new"     → save as filename_(1).ext or folder_(1)/
    """
    from utils.paths import dataset_uploads_root, ensure_dir

    filename = _sanitize_filename(file.filename or "upload")
    ext = Path(filename).suffix.lower()
    if ext not in _UPLOAD_EXTS:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: {', '.join(sorted(_UPLOAD_EXTS))}")

    upload_dir = dataset_uploads_root()
    ensure_dir(upload_dir)

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file uploaded")

    # ── Zip: extract to a named folder ──
    if ext == ".zip":
        stem = Path(filename).stem
        folder_path = upload_dir / stem

        if folder_path.exists():
            if action == "ask":
                return JSONResponse({"conflict": True, "filename": stem})
            elif action == "replace":
                shutil.rmtree(folder_path, ignore_errors=True)
            elif action == "new":
                counter = 1
                while folder_path.exists():
                    folder_path = upload_dir / f"{stem}_({counter})"
                    counter += 1

        folder_path = _extract_zip(content, folder_path.name, upload_dir)

        # Count data files inside
        all_files = list(folder_path.rglob("*.parquet")) + \
                    list(folder_path.rglob("*.json")) + \
                    list(folder_path.rglob("*.jsonl")) + \
                    list(folder_path.rglob("*.csv"))
        size_bytes = sum(f.stat().st_size for f in all_files if f.is_file())

        logger.info("Zip dataset extracted: %s → %s (%d data files)", filename, folder_path, len(all_files))
        return {
            "conflict": False,
            "filename": folder_path.name,
            "stored_path": str(folder_path),
            "base_dir": str(folder_path),
            "is_folder": True,
            "file_count": len(all_files),
            "size_bytes": size_bytes,
        }

    # ── Single file ──
    stored_path = upload_dir / filename

    if stored_path.exists():
        if action == "ask":
            return JSONResponse({"conflict": True, "filename": filename})
        elif action == "new":
            stem = Path(filename).stem
            counter = 1
            while stored_path.exists():
                new_name = f"{stem}_({counter}){ext}"
                stored_path = upload_dir / new_name
                counter += 1
            filename = stored_path.name
        # action == "replace" → fall through and overwrite

    stored_path.write_bytes(content)

    logger.info("Dataset uploaded: %s → %s (action=%s)", file.filename, stored_path, action)
    return {
        "conflict": False,
        "filename": filename,
        "stored_path": str(stored_path),
        "base_dir": None,
        "is_folder": False,
        "size_bytes": stored_path.stat().st_size,
    }


@router.delete("/upload/{filename:path}")
def delete_dataset(filename: str, _user=Depends(get_current_user)):
    """Delete an uploaded dataset file or extracted zip folder."""
    from utils.paths import dataset_uploads_root

    safe_name = _sanitize_filename(filename)
    target = dataset_uploads_root() / safe_name

    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
        logger.info("Dataset folder deleted: %s", safe_name)
        return {"deleted": safe_name}

    if not target.exists() or not target.is_file():
        raise HTTPException(404, "Dataset not found")
    if target.suffix.lower() not in (_UPLOAD_EXTS - {".zip"}):
        raise HTTPException(400, "Not an uploaded dataset file")
    target.unlink()
    logger.info("Dataset deleted: %s", safe_name)
    return {"deleted": safe_name}


@router.get("/local")
def list_local_datasets(_user=Depends(get_current_user)):
    """List local uploaded datasets and recipe-generated datasets."""
    from utils.paths import dataset_uploads_root, recipe_datasets_root

    items = []

    # Uploaded files and extracted zip folders
    uploads = dataset_uploads_root()
    if uploads.exists():
        for entry in sorted(uploads.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if entry.is_file() and entry.suffix.lower() in (_UPLOAD_EXTS - {".zip"}):
                items.append({
                    "id": entry.name,
                    "label": entry.name,
                    "path": str(entry),
                    "base_dir": None,
                    "is_folder": False,
                    "type": "upload",
                    "size_bytes": entry.stat().st_size,
                    "updated_at": entry.stat().st_mtime,
                    "rows": None,
                })
            elif entry.is_dir():
                # Extracted zip folder — check it has data files
                data_files = (list(entry.rglob("*.parquet")) +
                              list(entry.rglob("*.json")) +
                              list(entry.rglob("*.jsonl")) +
                              list(entry.rglob("*.csv")))
                if data_files:
                    size_bytes = sum(f.stat().st_size for f in data_files if f.is_file())
                    items.append({
                        "id": entry.name,
                        "label": entry.name,
                        "path": str(entry),
                        "base_dir": str(entry),
                        "is_folder": True,
                        "type": "vlm_folder",
                        "size_bytes": size_bytes,
                        "updated_at": entry.stat().st_mtime,
                        "rows": None,
                    })

    # Recipe-generated datasets
    recipes = recipe_datasets_root()
    if recipes.exists():
        for d in sorted(recipes.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir() or not d.name.startswith("recipe_"):
                continue
            parquet_dir = d / "parquet-files"
            if not parquet_dir.exists() or not any(parquet_dir.glob("*.parquet")):
                continue
            rows = None
            metadata_path = d / "metadata.json"
            if metadata_path.exists():
                try:
                    meta = json.loads(metadata_path.read_text())
                    rows = meta.get("actual_num_records") or meta.get("target_num_records")
                except Exception:
                    pass
            items.append({
                "id": d.name,
                "label": d.name,
                "path": str(parquet_dir),
                "type": "recipe",
                "size_bytes": None,
                "updated_at": d.stat().st_mtime,
                "rows": rows,
            })

    return {"datasets": items}


@router.post("/ai-assist-mapping")
def ai_assist_mapping(request: AiAssistRequest, _user=Depends(get_current_user)):
    """
    LLM-assisted column mapping.
    Tries unsloth-main's llm_conversion_advisor first.
    Falls back to heuristic column-role assignment.
    """
    # Heuristic column-role assignment (self-contained, no external deps)
    cols_lower = {c.lower(): c for c in request.columns}
    mapping = {}
    for role, candidates in [
        ("instruction", ["instruction", "question", "query", "input", "prompt", "human"]),
        ("input",       ["context", "background", "input"]),
        ("output",      ["output", "answer", "response", "assistant", "completion", "gpt"]),
    ]:
        for cand in candidates:
            if cand in cols_lower and cols_lower[cand] not in mapping.values():
                mapping[role] = cols_lower[cand]
                break

    if "instruction" in mapping and "output" in mapping:
        return {
            "success": True,
            "suggested_mapping": mapping,
            "dataset_type": "instruction-following",
            "source": "heuristic",
        }

    return {
        "success": False,
        "warning": "Could not determine column roles automatically. Please assign them manually.",
        "available_columns": request.columns,
    }

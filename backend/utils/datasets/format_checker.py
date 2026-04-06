"""
Dataset format detection — self-contained, no unsloth dependency.
Covers the most common fine-tuning dataset formats.
"""
from __future__ import annotations
import io
import base64
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Known format signatures ──
# Each entry: (format_name, required_columns, optional_columns)
_FORMAT_SIGNATURES = [
    # Alpaca-style
    ("alpaca",    ["instruction", "output"],         ["input"]),
    # ShareGPT conversations
    ("sharegpt",  ["conversations"],                 []),
    ("sharegpt",  ["messages"],                      []),
    # ChatML
    ("chatml",    ["messages"],                      ["system"]),
    # OpenAI-style
    ("openai",    ["messages"],                      []),
    # DPO/RLHF pairs
    ("dpo",       ["prompt", "chosen", "rejected"],  []),
    ("dpo",       ["question", "chosen", "rejected"],[]),
    # Raw text
    ("text",      ["text"],                          []),
    ("text",      ["content"],                       []),
    # QA pairs
    ("qa",        ["question", "answer"],            []),
    ("qa",        ["query", "response"],             []),
    # Completion style
    ("completion",["prompt", "completion"],          []),
    ("completion",["input", "output"],               []),
]


def detect_format_heuristic(columns: list[str]) -> Optional[str]:
    """
    Detect dataset format from column names alone.
    Returns format name or None if unknown.
    """
    cols_lower = {c.lower() for c in columns}
    for fmt, required, _ in _FORMAT_SIGNATURES:
        req_lower = {r.lower() for r in required}
        if req_lower.issubset(cols_lower):
            return fmt
    return None


def _detect_custom_format_heuristic(columns: list[str]) -> dict | None:
    """
    Keyword-scored heuristic for non-standard datasets.

    Maps columns to two simple roles: "input" and "output".
      input  — what the model receives (text, clause, question, image …)
      output — what the model should produce (label, answer, explanation …)

    Multiple columns can share the same role — the worker concatenates them.
    Returns None if we can't find at least one input AND one output column.
    """
    lower_to_orig = {c.lower(): c for c in columns}

    OUTPUT_WORDS = [
        "output", "answer", "response", "assistant", "completion",
        "label", "labels", "category", "class", "target",
        "explanation", "rationale", "reason", "result", "solution",
        "prediction", "expected", "reply",
    ]
    INPUT_WORDS = [
        "instruction", "question", "query", "input", "prompt", "human",
        "text", "sentence", "clause", "document", "passage", "content",
        "source", "problem", "request", "utterance",
        # NOTE: "image", "audio", "speech" intentionally excluded —
        # multimodal columns are detected separately and pinned as their own role,
        # not treated as text input.
    ]
    METADATA_EXACT = {
        "id", "idx", "index", "key", "split", "timestamp",
        "date", "url", "link", "score", "tag",
    }

    def _matches(col_lower: str, word_list: list[str]) -> bool:
        if col_lower in METADATA_EXACT:
            return False
        # exact keyword match
        if col_lower in word_list:
            return True
        # substring match
        return any(w in col_lower for w in word_list)

    mapping: dict[str, str] = {}
    assigned_input = False

    # Assign output first (more distinctive keywords), then input
    for col_lower, col_orig in lower_to_orig.items():
        if _matches(col_lower, OUTPUT_WORDS):
            mapping[col_orig] = "output"

    for col_lower, col_orig in lower_to_orig.items():
        if col_orig in mapping:
            continue
        if _matches(col_lower, INPUT_WORDS):
            mapping[col_orig] = "input"
            assigned_input = True

    has_input  = any(r == "input"  for r in mapping.values())
    has_output = any(r == "output" for r in mapping.values())

    if has_input and has_output:
        return mapping
    return None


def _get_columns(dataset) -> list[str]:
    if hasattr(dataset, "column_names"):
        return list(dataset.column_names)
    try:
        first = next(iter(dataset))
        return list(first.keys())
    except StopIteration:
        return []


def check_dataset_format(dataset, is_vlm: bool = False) -> dict:
    """
    Lightweight format check for the frontend.

    Returns:
        {
          requires_manual_mapping: bool,
          detected_format: str,
          columns: list[str],
          suggested_mapping: dict | None,
          is_image: bool,
          is_audio: bool,
          preview_samples: list | None  (first 3 rows, serialized)
        }
    """
    # ── Self-contained format detection ──
    columns = _get_columns(dataset)
    fmt = detect_format_heuristic(columns)

    # Check for multimodal columns
    IMAGE_COL_NAMES = {"image", "images", "img", "pixel_values"}
    is_image = any(c.lower() in IMAGE_COL_NAMES for c in columns)
    is_audio = any(c.lower() in {"audio", "speech", "waveform", "wav"} for c in columns)

    # Detect image column name and whether it is path-only (bytes=None)
    detected_image_column = next((c for c in columns if c.lower() in IMAGE_COL_NAMES), None)
    image_path_only = False
    if detected_image_column:
        try:
            sample_val = next(iter(dataset))[detected_image_column]
            if isinstance(sample_val, dict) and sample_val.get("bytes") is None and "path" in sample_val:
                image_path_only = True
        except Exception:
            pass

    # Build suggested_mapping for non-standard or ambiguous datasets
    suggested_mapping = None

    if fmt is None:
        # Unknown format — try heuristic on all columns
        suggested_mapping = _detect_custom_format_heuristic(columns)
        if suggested_mapping:
            fmt = "custom"

    elif fmt == "text" and len(columns) > 1:
        # "text" column exists but there are OTHER columns too (e.g. labels, explanation).
        # Override: run heuristic so user can map all columns properly.
        suggested_mapping = _detect_custom_format_heuristic(columns)
        if suggested_mapping:
            fmt = "custom"

    # Always remove the detected image column from suggested_mapping —
    # it is auto-pinned in the frontend as a locked "Image" role, not a text role.
    if detected_image_column and suggested_mapping and detected_image_column in suggested_mapping:
        del suggested_mapping[detected_image_column]

    requires_mapping = fmt is None

    return {
        "requires_manual_mapping": requires_mapping,
        "detected_format": fmt or "unknown",
        "columns": columns,
        "suggested_mapping": suggested_mapping,
        "is_image": is_image,
        "is_audio": is_audio,
        "image_path_only": image_path_only,
        "multimodal_columns": [detected_image_column] if detected_image_column else None,
        "detected_image_column": detected_image_column,
        "detected_audio_column": None,
        "detected_text_column": None,
        "detected_speaker_column": None,
        "warning": (
            "Images are stored as file paths (bytes=None). "
            "Make sure the image files are present on disk relative to the dataset folder."
            if image_path_only else None
        ),
        "preview_samples": serialize_preview_rows(dataset, max_rows=5),
    }


def serialize_preview_rows(dataset, max_rows: int = 10) -> list[dict]:
    """Serialize dataset rows to JSON-safe dicts."""
    rows = []
    try:
        for i, row in enumerate(dataset):
            if i >= max_rows:
                break
            rows.append(_serialize_row(row))
    except Exception as e:
        logger.debug("Error serializing rows: %s", e)
    return rows


def _serialize_value(value: Any) -> Any:
    """Make a single value JSON-safe."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # PIL Image
    try:
        from PIL.Image import Image as PILImage
        if isinstance(value, PILImage):
            buf = io.BytesIO()
            value.convert("RGB").save(buf, format="JPEG", quality=80)
            return {
                "type": "image", "mime": "image/jpeg",
                "width": value.width, "height": value.height,
                "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            }
    except Exception:
        pass
    if isinstance(value, dict):
        # HF image struct with bytes=None (path-only dataset) — return path string for preview
        if "bytes" in value and "path" in value and value.get("bytes") is None:
            return {"type": "image_path", "path": value.get("path") or ""}
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    return str(value)


def _serialize_row(row) -> dict:
    return {str(k): _serialize_value(v) for k, v in dict(row).items()}

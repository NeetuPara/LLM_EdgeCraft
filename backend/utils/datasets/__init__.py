"""
Dataset utilities — format detection and conversion.
Provides a simple built-in implementation for common formats,
then tries unsloth-main's richer implementation when available.
"""
from utils.datasets.format_checker import (
    check_dataset_format,
    detect_format_heuristic,
    serialize_preview_rows,
)

__all__ = ["check_dataset_format", "detect_format_heuristic", "serialize_preview_rows"]

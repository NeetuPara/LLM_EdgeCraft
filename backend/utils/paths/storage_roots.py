"""
All canonical paths for UnslothCraft.
Adapted from unsloth-main/studio/backend/utils/paths/storage_roots.py
Change: ~/.unsloth/studio/ → ~/.unslothcraft/
"""
import os
import tempfile
from pathlib import Path


# ── Root ──
def base_root() -> Path:
    return Path.home() / ".unslothcraft"

def cache_root() -> Path:
    return base_root() / "cache"

def assets_root() -> Path:
    return base_root() / "assets"

def datasets_root() -> Path:
    return assets_root() / "datasets"

def dataset_uploads_root() -> Path:
    return datasets_root() / "uploads"

def recipe_datasets_root() -> Path:
    return datasets_root() / "recipes"

def outputs_root() -> Path:
    return base_root() / "outputs"

def exports_root() -> Path:
    return base_root() / "exports"

def auth_root() -> Path:
    return base_root() / "auth"

def studio_db_path() -> Path:
    return base_root() / "unslothcraft.db"

def tmp_root() -> Path:
    return Path(tempfile.gettempdir()) / "unslothcraft"

def tensorboard_root() -> Path:
    return base_root() / "runs"

def seed_uploads_root() -> Path:
    return datasets_root() / "seed-uploads"

def unstructured_uploads_root() -> Path:
    return datasets_root() / "unstructured-uploads"

# ── HuggingFace cache ──
def hf_default_cache_dir() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"

def legacy_hf_cache_dir() -> Path:
    return cache_root() / "huggingface" / "hub"


# ── LM Studio model dirs ──
def lmstudio_model_dirs() -> list[Path]:
    """Discover LM Studio model directories."""
    candidates = [
        Path.home() / ".lmstudio" / "models",
        Path.home() / ".cache" / "lm-studio" / "models",
    ]
    # Try settings.json
    settings = Path.home() / ".lmstudio" / "settings.json"
    if settings.exists():
        try:
            import json
            data = json.loads(settings.read_text())
            dl = data.get("downloadsFolder")
            if dl:
                candidates.insert(0, Path(dl))
        except Exception:
            pass

    seen: set[Path] = set()
    result: list[Path] = []
    for p in candidates:
        rp = p.resolve() if p.exists() else p
        if rp not in seen and p.exists():
            seen.add(rp)
            result.append(p)
    return result


# ── Helpers ──
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_base_directories() -> None:
    """Create all required directories at startup."""
    for fn in [
        base_root, assets_root, datasets_root, dataset_uploads_root,
        recipe_datasets_root, unstructured_uploads_root, outputs_root,
        exports_root, auth_root, tensorboard_root,
    ]:
        fn().mkdir(parents=True, exist_ok=True)


def resolve_output_dir(path_value: str | Path) -> Path:
    """Resolve an output directory path (absolute or relative under outputs_root)."""
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    # Strip leading "outputs" segment if present
    parts = p.parts
    if parts and parts[0] in ("outputs", "output"):
        p = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return outputs_root() / p


def resolve_export_dir(path_value: str | Path) -> Path:
    """Resolve an export directory path."""
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] in ("exports", "export"):
        p = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return exports_root() / p

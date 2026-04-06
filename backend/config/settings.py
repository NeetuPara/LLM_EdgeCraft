import os
from pathlib import Path

# ── Auth ──
JWT_SECRET = os.getenv("JWT_SECRET", "unslothcraft-dev-secret-change-in-production")

# ── Database ──
# Default: ~/.unslothcraft/unslothcraft.db (SQLite, no server needed)
_db_dir = Path.home() / ".unslothcraft"
_db_dir.mkdir(parents=True, exist_ok=True)
_default_db = f"sqlite:///{_db_dir / 'unslothcraft.db'}"
DATABASE_URL = os.getenv("DATABASE_URL", _default_db)

# ── CORS ──
_cors_raw = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5174,http://127.0.0.1:5174,http://localhost:5173,http://127.0.0.1:5173",
)
CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]

# ── Server ──
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5174")

# ── Cookie ──
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")

# ── HuggingFace ──
HF_TOKEN = os.getenv("HF_TOKEN", "")

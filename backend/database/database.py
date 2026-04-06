# Adapted from EdgeCraft — SQLite instead of PostgreSQL
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import DATABASE_URL

logger = logging.getLogger(__name__)

# SQLite needs check_same_thread=False for multi-threaded FastAPI
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    import database.models  # noqa: F401 — registers all models with Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified at: %s", DATABASE_URL)

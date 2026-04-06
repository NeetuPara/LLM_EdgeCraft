# Adapted from EdgeCraft — User model only, SQLite-compatible types
from sqlalchemy import Column, Integer, String
from sqlalchemy.sql import func
from sqlalchemy import DateTime
from database.database import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email         = Column(String, unique=True, nullable=False, index=True)
    name          = Column(String, nullable=True)
    password_hash = Column(String, nullable=False)
    role          = Column(String, default="user", nullable=False)
    created_at    = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<User id={self.id} email={self.email}>"

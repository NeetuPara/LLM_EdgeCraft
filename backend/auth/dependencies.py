# Ported from EdgeCraft (Edge_ML_Platform-dev/backend/auth/dependencies.py)
import logging
from fastapi import Header, HTTPException, Depends, Cookie
from sqlalchemy.orm import Session
from database.database import get_db
from auth.jwt_handler import decode_token
from database.models import User
from typing import Optional

logger = logging.getLogger(__name__)


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    access_token: Optional[str] = Cookie(default=None),
    db: Session = Depends(get_db),
) -> User:
    """
    Accepts token from either:
    1. Authorization header: "Bearer <token>"
    2. HTTP-only cookie: access_token
    """
    token = None

    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
    elif access_token:
        token = access_token

    if not token:
        raise HTTPException(401, "Not authenticated. Please log in.")

    try:
        payload = decode_token(token)
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(401, "Token is missing user_id claim")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Token decode error: %s", e)
        raise HTTPException(401, "Invalid or expired token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(401, "User account not found")

    return user


def get_current_user_id(
    current_user: User = Depends(get_current_user),
) -> int:
    return current_user.id

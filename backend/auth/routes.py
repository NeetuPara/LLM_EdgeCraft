# Adapted from EdgeCraft (Edge_ML_Platform-dev/backend/auth/routes.py)
# Changes: email-based auth, no Microsoft SSO, no HTTPOnly cookie (SPA uses Bearer token)
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from database.database import get_db
from database.models import User
from auth.hashing import hash_password, verify_password
from auth.dependencies import get_current_user
from auth.jwt_handler import create_token
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Schemas ──
class SignupReq(BaseModel):
    email: str
    password: str
    name: str


class LoginReq(BaseModel):
    email: str
    password: str


class ChangePasswordReq(BaseModel):
    current_password: str
    new_password: str


# ── Routes ──
@router.post("/signup")
def signup(data: SignupReq, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email.lower().strip()).first():
        raise HTTPException(400, "An account with this email already exists.")

    user = User(
        email=data.email.lower().strip(),
        name=data.name.strip(),
        password_hash=hash_password(data.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_token({"user_id": user.id, "role": user.role})
    return {
        "msg": "Account created",
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email, "name": user.name},
    }


@router.post("/login")
def login(data: LoginReq, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email.lower().strip()).first()

    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password.")

    token = create_token({"user_id": user.id, "role": user.role})
    return {
        "msg": "Logged in",
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email, "name": user.name},
    }


@router.post("/logout")
def logout():
    return {"msg": "Logged out"}


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
    }


@router.post("/change-password")
def change_password(
    data: ChangePasswordReq,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(data.current_password, user.password_hash):
        raise HTTPException(400, "Current password is incorrect.")

    user.password_hash = hash_password(data.new_password)
    db.commit()
    return {"msg": "Password updated"}

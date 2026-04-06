# Ported from EdgeCraft (Edge_ML_Platform-dev/backend/auth/jwt_handler.py)
from jose import jwt
from datetime import datetime, timedelta
from config.settings import JWT_SECRET

SECRET = JWT_SECRET
ALGO = "HS256"

def create_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=1)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET, algorithm=ALGO)

def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET, algorithms=[ALGO])

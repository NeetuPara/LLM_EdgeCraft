# PBKDF2-HMAC-SHA256 — stdlib only, no bcrypt dependency issues
# Format stored in DB: "pbkdf2$<salt_hex>$<hash_hex>"
import hashlib
import hmac
import os
import secrets

_ITERATIONS = 260_000
_ALGORITHM = "sha256"


def hash_password(password: str) -> str:
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac(_ALGORITHM, password.encode(), salt, _ITERATIONS)
    return f"pbkdf2${salt.hex()}${key.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        _, salt_hex, hash_hex = stored_hash.split("$", 2)
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac(_ALGORITHM, password.encode(), salt, _ITERATIONS)
        return hmac.compare_digest(key.hex(), hash_hex)
    except Exception:
        return False

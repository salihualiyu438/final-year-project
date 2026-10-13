"""Lightweight local authentication for the surveillance web application.

Uses only Python's standard library so Phase 5 does not add a mandatory
external authentication service. Passwords are verified with scrypt and
sessions are signed with an HMAC-protected cookie.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Optional

SESSION_COOKIE = "surveillance_session"
# Set to 1 for HTTPS remote deployment (e.g. Cloudflare Tunnel).
# Keep 0 for local http://127.0.0.1 development.
SECURE_COOKIE = os.getenv("SURVEILLANCE_SECURE_COOKIE", "0").strip().lower() in {"1", "true", "yes", "on"}
SESSION_TTL_S = int(os.getenv("SESSION_TTL_S", "28800"))  # 8 hours
SESSION_SECRET = os.getenv("SURVEILLANCE_SESSION_SECRET", "")

# Development defaults. Change these before remote deployment.
DEFAULT_USERS = {
    "operator": {"role": "operator", "password": os.getenv("SURVEILLANCE_OPERATOR_PASSWORD", "operator123")},
    "owner": {"role": "owner", "password": os.getenv("SURVEILLANCE_OWNER_PASSWORD", "owner123")},
}

if not SESSION_SECRET:
    # Stable for this process, but deliberately not a production secret.
    SESSION_SECRET = secrets.token_urlsafe(32)
    print("[AUTH] No SURVEILLANCE_SESSION_SECRET set; generated a temporary session secret for this run.")


def _sign(payload: str) -> str:
    return hmac.new(SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _encode(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":"), sort_keys=True).encode()
    payload = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return payload + "." + _sign(payload)


def _decode(token: str) -> Optional[dict]:
    try:
        payload, sig = token.rsplit(".", 1)
        if not hmac.compare_digest(sig, _sign(payload)):
            return None
        padded = payload + "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded).decode())
        if int(data.get("exp", 0)) < int(time.time()):
            return None
        return data
    except Exception:
        return None


def _password_digest(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


def verify_password(username: str, password: str) -> Optional[dict]:
    user = DEFAULT_USERS.get(username)
    if not user:
        return None
    # Per-process deterministic salt derived from the username and session secret.
    # This is sufficient for this local project; replace with a persistent user
    # database before production deployment.
    salt = hashlib.sha256(("surveillance:" + username + ":" + SESSION_SECRET).encode()).digest()[:16]
    expected = _password_digest(user["password"], salt)
    supplied = _password_digest(password, salt)
    if hmac.compare_digest(expected, supplied):
        return {"username": username, "role": user["role"]}
    return None


def create_session(user: dict) -> str:
    now = int(time.time())
    return _encode({
        "username": user["username"],
        "role": user["role"],
        "iat": now,
        "exp": now + SESSION_TTL_S,
    })


def get_session_from_cookie(token: Optional[str]) -> Optional[dict]:
    return _decode(token) if token else None

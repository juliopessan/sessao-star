"""Small, local-first authentication layer for STAR Session.

Passwords are never stored directly. Sessions use opaque, expiring cookies whose
hashes are stored in SQLite, keeping the browser token out of the database.
"""

import hashlib
import hmac
import secrets
import sqlite3
import time
from typing import Optional

import db

SESSION_COOKIE = "star_session"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30
PASSWORD_SCHEME = "scrypt"


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _password_bytes(password: str) -> bytes:
    return (password or "").encode("utf-8")


def hash_password(password: str) -> str:
    raw = _password_bytes(password)
    if len(raw) < 8:
        raise ValueError("Password must contain at least 8 characters.")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(raw, salt=salt, n=2**14, r=8, p=1)
    return f"{PASSWORD_SCHEME}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = encoded.split("$", 2)
        if scheme != PASSWORD_SCHEME:
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(
            _password_bytes(password),
            salt=bytes.fromhex(salt_hex),
            n=2**14,
            r=8,
            p=1,
        )
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError):
        return False


def _public_user(row) -> dict:
    return {"id": row["id"], "name": row["name"], "email": row["email"]}


def create_user(name: str, email: str, password: str) -> dict:
    clean_name = (name or "").strip()
    clean_email = normalize_email(email)
    if not clean_name:
        raise ValueError("Name is required.")
    if "@" not in clean_email or "." not in clean_email.split("@")[-1]:
        raise ValueError("Enter a valid email address.")

    password_hash = hash_password(password)
    conn = db.get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO users (created_at, name, email, password_hash) VALUES (?, ?, ?, ?)",
            (time.time(), clean_name, clean_email, password_hash),
        )
        user_id = cur.lastrowid
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _public_user(row)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise ValueError("An account with this email already exists.") from exc
    finally:
        conn.close()


def authenticate(email: str, password: str) -> Optional[dict]:
    conn = db.get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        (normalize_email(email),),
    ).fetchone()
    conn.close()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return _public_user(row)


def claim_legacy_sessions(user_id: int) -> None:
    """Attach pre-auth local history to the first authenticated account."""
    conn = db.get_conn()
    conn.execute("UPDATE sessions SET user_id = ? WHERE user_id IS NULL", (user_id,))
    conn.commit()
    conn.close()


def create_session(user_id: int) -> str:
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    now = time.time()
    conn = db.get_conn()
    conn.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (now,))
    conn.execute(
        "INSERT INTO auth_sessions (user_id, token_hash, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (user_id, token_hash, now, now + SESSION_TTL_SECONDS),
    )
    conn.commit()
    conn.close()
    return raw_token


def user_from_token(raw_token: Optional[str]) -> Optional[dict]:
    if not raw_token:
        return None
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    conn = db.get_conn()
    row = conn.execute(
        """SELECT u.id, u.name, u.email
           FROM auth_sessions a
           JOIN users u ON u.id = a.user_id
           WHERE a.token_hash = ? AND a.expires_at > ?""",
        (token_hash, time.time()),
    ).fetchone()
    conn.close()
    return _public_user(row) if row else None


def revoke_session(raw_token: Optional[str]) -> None:
    if not raw_token:
        return
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    conn = db.get_conn()
    conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))
    conn.commit()
    conn.close()

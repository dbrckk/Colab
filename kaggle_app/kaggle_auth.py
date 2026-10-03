from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any


def credentials_ready(username: str, api_token: str, legacy_key: str) -> bool:
    return bool((username or "").strip() and ((api_token or "").strip() or (legacy_key or "").strip()))


def credential_fingerprint(username: str, secret: str) -> str:
    username = (username or "").strip()
    secret = (secret or "").strip()
    if not username or not secret:
        return ""
    return hashlib.sha256(f"{username}\0{secret}".encode("utf-8")).hexdigest()


def auth_cache_valid(
    cache: dict[str, Any],
    fingerprint: str,
    *,
    now: float,
    max_age_seconds: int = 300,
) -> bool:
    if not fingerprint:
        return False
    return bool(
        cache.get("fingerprint") == fingerprint
        and now - float(cache.get("ts") or 0) <= max(0, int(max_age_seconds))
    )


def validate_credential_fields(username: str, api_token: str, legacy_key: str) -> None:
    username = (username or "").strip()
    api_token = (api_token or "").strip()
    legacy_key = (legacy_key or "").strip()

    if not username:
        raise ValueError("KAGGLE_USERNAME est requis pour créer les kernels/datasets.")
    if not api_token and not legacy_key:
        raise ValueError("Ajoute KAGGLE_API_TOKEN (recommandé) ou l'ancien KAGGLE_KEY.")

    def _safe_env_value(name: str, value: str) -> None:
        if any(ch in value for ch in ("\n", "\r", "\x00")):
            raise ValueError(f"{name} contient un caractère interdit.")
        if len(value) > 4096:
            raise ValueError(f"{name} est anormalement long.")

    _safe_env_value("KAGGLE_USERNAME", username)
    _safe_env_value("KAGGLE_API_TOKEN", api_token)
    _safe_env_value("KAGGLE_KEY", legacy_key)

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", username):
        raise ValueError("KAGGLE_USERNAME contient des caractères non valides.")


CREDENTIAL_KEYS = ("KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY")


def write_credential_env(
    env_path: Path,
    *,
    username: str,
    api_token: str,
    legacy_key: str,
) -> None:
    preserved: list[str] = []
    credential_key_set = set(CREDENTIAL_KEYS)

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
            if key not in credential_key_set:
                preserved.append(line)

    lines = preserved + [f"KAGGLE_USERNAME={username}"]
    if api_token:
        lines.append(f"KAGGLE_API_TOKEN={api_token}")
    else:
        lines.append(f"KAGGLE_KEY={legacy_key}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(lines).rstrip() + "\n"
    tmp_env = env_path.with_name(f".{env_path.name}.{uuid.uuid4().hex[:8]}.part")

    try:
        tmp_env.write_text(payload, encoding="utf-8")
        try:
            os.chmod(tmp_env, 0o600)
        except Exception:
            pass
        os.replace(tmp_env, env_path)
        try:
            os.chmod(env_path, 0o600)
        except Exception:
            pass
    finally:
        try:
            tmp_env.unlink(missing_ok=True)
        except OSError:
            pass

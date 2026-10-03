from __future__ import annotations

import hashlib
import re
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

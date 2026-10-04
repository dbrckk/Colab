from __future__ import annotations

from typing import Any


TERMINAL_STATES = {"done", "error", "cancelled", "interrupted"}
ACTIVE_STATES = {
    "preparing",
    "uploading_inputs",
    "submitting",
    "queued",
    "waiting_auth",
    "running",
    "recovering",
    "downloading",
    "cancel_requested",
}


def terminal_status_message(job: dict[str, Any]) -> str | None:
    status = str(job.get("status") or "")
    if status in TERMINAL_STATES:
        return f"Job déjà terminé: {status}"
    return None


def retry_block_reason(job: dict[str, Any]) -> str | None:
    status = str(job.get("status") or "")
    if status not in ACTIVE_STATES:
        return None
    if status == "waiting_auth":
        if job.get("kernel_ref"):
            return (
                "Ce job attend l'authentification pour reprendre son kernel Kaggle existant. "
                "Configure les identifiants au lieu de le relancer."
            )
        return (
            "Ce job attend l'authentification pour reprendre la file locale. "
            "Configure les identifiants au lieu de créer un doublon."
        )
    return "Ce job est encore actif."


def can_cancel_locally(
    *,
    kernel_ref: str,
    future_active: bool,
    cancelled_before_start: bool,
) -> bool:
    return bool(
        not kernel_ref
        and (cancelled_before_start or not future_active)
    )

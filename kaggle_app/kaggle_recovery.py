from __future__ import annotations

from typing import Any


REMOTE_ACTIVE_STATES = {"submitting", "queued", "running", "recovering", "downloading"}
LOCAL_REPLAY_STATES = {"preparing", "uploading_inputs"}
RECOVERABLE_ERROR_STATES = {"error", "interrupted"}
MAX_AUTO_RECOVERY_ATTEMPTS = 3


def auto_recovery_attempts(row: dict[str, Any]) -> int:
    meta = row.get("meta") or {}
    return int(meta.get("auto_recovery_attempts") or 0)


def startup_recovery_action(
    row: dict[str, Any],
    credentials_ready: bool,
) -> str:
    status = str(row.get("status") or "")
    kernel_ref = str(row.get("kernel_ref") or "")
    meta = row.get("meta") or {}

    if status in LOCAL_REPLAY_STATES:
        return "resume_local" if credentials_ready else "wait_local_auth"

    if status in REMOTE_ACTIVE_STATES:
        if kernel_ref:
            return "resume_remote" if credentials_ready else "wait_remote_auth"
        return "resume_local" if credentials_ready else "wait_local_auth"

    if status in {"queued", "waiting_auth"} and not kernel_ref:
        return "resume_local" if credentials_ready else "wait_local_auth"

    if status == "waiting_auth" and kernel_ref:
        return "resume_remote" if credentials_ready else "wait_remote_auth"

    if status in RECOVERABLE_ERROR_STATES and kernel_ref:
        if not meta.get("recover_outputs_available"):
            return "none"
        if auto_recovery_attempts(row) >= MAX_AUTO_RECOVERY_ATTEMPTS:
            return "none"
        return "resume_recoverable" if credentials_ready else "none"

    return "none"


def keep_remote_kernel_for_retry(
    status: str,
    meta: dict[str, Any],
    kernel_ref: str,
) -> bool:
    return bool(
        kernel_ref
        and status in {"error", "waiting_auth"}
        and meta.get("recover_outputs_available")
    )


def recoverable_output_candidate(row: dict[str, Any]) -> bool:
    if row.get("status") not in RECOVERABLE_ERROR_STATES:
        return False
    if not row.get("kernel_ref"):
        return False
    meta = row.get("meta") or {}
    if not meta.get("recover_outputs_available"):
        return False
    return auto_recovery_attempts(row) < MAX_AUTO_RECOVERY_ATTEMPTS

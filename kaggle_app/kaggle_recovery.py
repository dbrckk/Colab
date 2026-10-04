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

    if (
        status == "submitting"
        and kernel_ref
        and meta.get("remote_submission_confirmed") is False
    ):
        return "probe_submission" if credentials_ready else "wait_submission_auth"

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


def recovery_marked(row: dict[str, Any]) -> bool:
    return bool((row.get("meta") or {}).get("recover_outputs_available"))


def has_recoverable_outputs(row: dict[str, Any]) -> bool:
    return bool(row.get("kernel_ref") and recovery_marked(row))


def remote_kernel_is_preserved(row: dict[str, Any]) -> bool:
    if not row.get("kernel_ref"):
        return False
    status = row.get("status")
    if status in {"waiting_auth", "recovering", "downloading"}:
        return True
    return bool(
        status in RECOVERABLE_ERROR_STATES
        and (row.get("meta") or {}).get("recover_outputs_available")
    )


def classify_remote_recovery_failure(
    *,
    auth_required: bool,
    missing_remote: bool,
    remote_failed: bool,
) -> tuple[str, bool, str, bool]:
    if auth_required:
        return "waiting_auth", True, "auth_required", False
    if missing_remote:
        return "error", False, "remote_missing", False
    if remote_failed:
        return "error", False, "remote_failed", True
    return "error", True, "downloading", False


def preserve_kernel_after_execute_failure(
    *,
    kernel_ref: str,
    failed_phase: str,
    remote_failed: bool,
) -> bool:
    return bool(
        kernel_ref
        and failed_phase in {
            "submitting", "queued", "running", "downloading", "recovering",
        }
        and not remote_failed
    )


def expired_recovery_candidates(
    rows: list[dict[str, Any]],
    *,
    now: float,
    retention_days: int,
) -> list[dict[str, Any]]:
    cutoff = now - max(1, int(retention_days)) * 86400
    return [
        row
        for row in rows
        if recovery_marked(row)
        and float(row.get("updated_at") or 0) < cutoff
    ]


def submission_confirmation_pending(row: dict[str, Any]) -> bool:
    return bool(
        row.get("kernel_ref")
        and (row.get("meta") or {}).get("remote_submission_confirmed") is False
    )

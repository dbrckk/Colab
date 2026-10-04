from __future__ import annotations

from typing import Any


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


def format_storage_bytes(total_bytes: int) -> str:
    total_bytes = max(0, int(total_bytes))
    if total_bytes >= 1024**3:
        return f"{total_bytes / 1024**3:.2f} Go"
    return f"{total_bytes / 1024**2:.1f} Mo"


def dashboard_summary_text(
    jobs: list[dict[str, Any]],
    *,
    total_bytes: int,
    auth_label: str,
) -> str:
    active = sum(1 for job in jobs if job.get("status") in ACTIVE_STATES)
    done = sum(1 for job in jobs if job.get("status") == "done")
    failed = sum(1 for job in jobs if job.get("status") in {"error", "interrupted"})
    cancelled = sum(1 for job in jobs if job.get("status") == "cancelled")
    storage = format_storage_bytes(total_bytes)

    return (
        f"**{auth_label}**  •  "
        f"Actifs **{active}**  •  Terminés **{done}**  •  "
        f"Erreurs **{failed}**  •  Annulés **{cancelled}**  •  "
        f"Stockage **{storage}**"
    )

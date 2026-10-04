from __future__ import annotations

from typing import Any, Callable


RowPredicate = Callable[[dict[str, Any]], bool]


def recovery_diagnostic_lines(
    jobs: list[dict[str, Any]],
    *,
    recovery_marked: RowPredicate,
    remote_kernel_is_preserved: RowPredicate,
    submission_confirmation_pending: RowPredicate,
) -> list[str]:
    waiting_auth = sum(1 for row in jobs if row.get("status") == "waiting_auth")
    recoverable = sum(1 for row in jobs if recovery_marked(row))
    remote_preserved = sum(1 for row in jobs if remote_kernel_is_preserved(row))
    unconfirmed_submissions = sum(
        1 for row in jobs if submission_confirmation_pending(row)
    )

    lines: list[str] = []
    if waiting_auth:
        lines.append(f"⚠️ {waiting_auth} job(s) attendent l'authentification.")
    if recoverable:
        lines.append(
            f"♻️ {recoverable} job(s) ont des outputs Kaggle récupérables sans recalcul."
        )
    if remote_preserved:
        lines.append(
            f"🛰️ {remote_preserved} kernel(s) distant(s) conservé(s) pour reprise."
        )
    if unconfirmed_submissions:
        lines.append(
            f"🔎 {unconfirmed_submissions} soumission(s) Kaggle distante(s) restent à confirmer avant tout replay."
        )
    if (
        not waiting_auth
        and not recoverable
        and not remote_preserved
        and not unconfirmed_submissions
    ):
        lines.append("✅ Aucun job de reprise en attente.")
    return lines

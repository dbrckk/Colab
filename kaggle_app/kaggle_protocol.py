from __future__ import annotations

import re


KERNEL_TERMINAL_OK = ("complete", "completed")
KERNEL_TERMINAL_BAD = ("error", "failed", "cancelled", "canceled")


def is_auth_cli_error(text: str) -> bool:
    low = (text or "").lower()

    direct_markers = (
        "401",
        "unauthorized",
        "authentication required",
        "invalid token",
        "invalid api token",
        "credentials are required",
        "missing credentials",
        "authentication failed",
    )
    if any(marker in low for marker in direct_markers):
        return True

    permission_markers = ("403", "forbidden")
    credential_context = (
        "token",
        "credential",
        "authentication",
        "api key",
    )
    return (
        any(marker in low for marker in permission_markers)
        and any(marker in low for marker in credential_context)
    )


def is_transient_cli_error(text: str) -> bool:
    low = (text or "").lower()
    transient = (
        "timed out", "timeout", "temporarily unavailable", "connection reset",
        "connection aborted", "connection refused", "remote disconnected",
        "server error", "bad gateway", "gateway timeout", "service unavailable",
        "too many requests", "rate limit", "429", "500", "502", "503", "504",
    )
    permanent = (
        "401", "403", "unauthorized", "forbidden", "invalid token",
        "authentication", "usage:", "unrecognized arguments", "invalid choice",
    )
    if any(token in low for token in permanent):
        return False
    return any(token in low for token in transient)


def remote_kernel_missing(text: str) -> bool:
    low = (text or "").lower()
    return any(token in low for token in (
        "404", "not found", "does not exist", "could not find",
        "kernel not found", "no such kernel",
    ))


def parse_dataset_status(output: str) -> str:
    match = re.search(
        r"(?im)^\s*(?:dataset\s+)?status\s*[:=]\s*"
        r"(ready|complete|completed|creating|pending|queued|error|failed)\s*$",
        output or "",
    )
    token = match.group(1).lower() if match else ""
    if not token:
        standalone = {
            line.strip().lower()
            for line in (output or "").splitlines()
            if line.strip().lower() in {
                "ready", "complete", "completed", "creating",
                "pending", "queued", "error", "failed",
            }
        }
        if len(standalone) == 1:
            token = next(iter(standalone))

    groups = {
        "ready": "ready", "complete": "ready", "completed": "ready",
        "creating": "pending", "pending": "pending", "queued": "pending",
        "error": "error", "failed": "error",
    }
    mentioned = {
        groups[t.lower()]
        for t in re.findall(
            r"(?i)\b(ready|complete|completed|creating|pending|queued|error|failed)\b",
            output or "",
        )
    }
    if len(mentioned) > 1:
        return "unknown"
    if token in {"ready", "complete", "completed"}:
        return "ready"
    if token in {"creating", "pending", "queued"}:
        return "pending"
    if token in {"error", "failed"}:
        return "error"
    return "unknown"


def parse_kernel_status(output: str) -> str:
    match = re.search(
        r"(?im)^\s*(?:kernel\s+)?status\s*[:=]\s*"
        r"(queued|pending|running|active|executing|complete|completed|error|failed|cancelled|canceled)\s*$",
        output or "",
    )
    state_token = match.group(1).lower() if match else ""
    if not state_token:
        standalone_states = {
            line.strip().lower()
            for line in (output or "").splitlines()
            if line.strip().lower() in {
                "queued", "pending", "running", "active", "executing",
                "complete", "completed", "error", "failed",
                "cancelled", "canceled",
            }
        }
        if len(standalone_states) == 1:
            state_token = next(iter(standalone_states))

    state_group = {
        "queued": "queued", "pending": "queued",
        "running": "running", "active": "running", "executing": "running",
        "complete": "complete", "completed": "complete",
        "error": "error", "failed": "error",
        "cancelled": "error", "canceled": "error",
    }
    mentioned = {
        state_group[token.lower()]
        for token in re.findall(
            r"(?i)\b(queued|pending|running|active|executing|complete|completed|error|failed|cancelled|canceled)\b",
            output or "",
        )
    }
    if len(mentioned) > 1:
        raise RuntimeError(
            "Statut Kaggle ambigu; plusieurs états incompatibles ont été détectés. "
            "Le kernel distant est conservé pour reprise: " + (output or "")[-1500:]
        )

    if state_token in KERNEL_TERMINAL_BAD:
        return "error"
    if state_token in KERNEL_TERMINAL_OK:
        return "complete"
    if state_token in {"queued", "pending"}:
        return "queued"
    if state_token in {"running", "active", "executing"}:
        return "running"
    raise RuntimeError(
        "Statut Kaggle non reconnu; le kernel distant est conservé pour reprise: "
        + (output or "")[-1500:]
    )

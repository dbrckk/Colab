from __future__ import annotations

from pathlib import Path
from typing import Any


def has_valid_local_media(artifacts: list[dict[str, Any]]) -> bool:
    return any(
        artifact.get("kind") in {"image", "video"}
        and Path(artifact.get("path") or "").is_file()
        for artifact in artifacts
    )


def classify_missing_local_outputs(
    *,
    kernel_ref: str,
    kernel_state: str,
) -> tuple[str, bool, str]:
    if not kernel_ref:
        return (
            "no_remote",
            False,
            "Résultats locaux manquants et aucun kernel Kaggle n'est disponible pour récupération.",
        )
    if kernel_state == "complete":
        return (
            "recoverable",
            True,
            "Résultats locaux manquants; outputs Kaggle encore disponibles.",
        )
    if kernel_state == "error":
        return (
            "remote_failed",
            False,
            "Résultats locaux manquants et le kernel Kaggle n'est plus récupérable.",
        )
    return (
        "unknown",
        False,
        "Résultats locaux manquants. L'état du kernel Kaggle n'a pas pu être confirmé; "
        "réessaie le diagnostic ou relance le job.",
    )

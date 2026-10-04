from __future__ import annotations

from pathlib import Path
from typing import Any


def stale_artifact_ids(rows: list[dict[str, Any]]) -> list[int]:
    stale: list[int] = []
    for row in rows:
        path = row.get("path") or ""
        if not path or not Path(path).is_file():
            stale.append(int(row["id"]))
    return stale

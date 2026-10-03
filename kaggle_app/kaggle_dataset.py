from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def write_dataset_bundle(
    folder: Path,
    *,
    job_id: str,
    dataset_ref: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    staged_config = dict(config)

    for key in ("source_image", "target_video"):
        src = staged_config.get(key)
        if not src:
            continue

        path = Path(src)
        if not path.exists():
            raise FileNotFoundError(f"Fichier d'entrée introuvable: {path}")

        target_name = (
            f"source_image{path.suffix}"
            if key == "source_image"
            else f"target_video{path.suffix}"
        )
        target = folder / target_name
        shutil.copy2(path, target)
        staged_config[key] = target.name

    (folder / "job_config.json").write_text(
        json.dumps(staged_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "title": f"Qwen input {job_id}"[:50],
        "id": dataset_ref,
        "licenses": [{"name": "other"}],
    }
    (folder / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return staged_config

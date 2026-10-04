from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable


HashFile = Callable[[Path], str]


def build_job_archive(
    *,
    export_root: Path,
    job_id: str,
    job: dict[str, Any],
    artifacts: list[dict[str, Any]],
    hash_file: HashFile,
    exported_at: str,
) -> str:
    export_root.mkdir(parents=True, exist_ok=True)
    work = export_root / f".{job_id}-export"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    try:
        media_dir = work / "media"
        media_dir.mkdir()
        exported_artifacts: list[dict[str, Any]] = []

        for artifact in artifacts:
            src = Path(artifact.get("path") or "")
            if not src.exists() or not src.is_file():
                continue

            target = media_dir / src.name
            shutil.copy2(src, target)
            exported_artifacts.append(
                {
                    "name": target.name,
                    "kind": artifact.get("kind") or "file",
                    "size": int(target.stat().st_size),
                    "sha256": hash_file(target),
                }
            )

        manifest = {
            "job": job,
            "artifacts": artifacts,
            "exported_artifacts": exported_artifacts,
            "exported_at": exported_at,
        }
        (work / "job.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        archive_base = export_root / job_id
        return shutil.make_archive(str(archive_base), "zip", root_dir=work)
    finally:
        shutil.rmtree(work, ignore_errors=True)

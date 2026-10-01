from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"}

def kind_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXT:
        return "image"
    if suffix in VIDEO_EXT:
        return "video"
    return "file"

def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def import_outputs(job_id: str, source_dir: Path, storage_root: Path) -> list[tuple[Path, str]]:
    dest = storage_root / job_id
    dest.mkdir(parents=True, exist_ok=True)
    imported: list[tuple[Path, str]] = []
    seen: set[str] = set()

    existing_by_digest: dict[str, Path] = {}
    for current in dest.iterdir():
        if current.is_file():
            try:
                existing_by_digest[_sha1(current)] = current
            except Exception:
                pass

    for src in sorted(source_dir.rglob("*")):
        if not src.is_file() or src.name.startswith("."):
            continue
        digest = _sha1(src)
        if digest in seen:
            continue
        seen.add(digest)

        existing = existing_by_digest.get(digest)
        if existing is not None:
            imported.append((existing, kind_for(existing)))
            continue

        target = dest / src.name
        if target.exists():
            target = dest / f"{target.stem}_{digest[:10]}{target.suffix}"
        shutil.copy2(src, target)
        existing_by_digest[digest] = target
        imported.append((target, kind_for(target)))
    return imported

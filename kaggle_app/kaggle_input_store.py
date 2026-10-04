from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_input_store(storage_root: Path) -> Path:
    root = storage_root / "_input_store"
    root.mkdir(parents=True, exist_ok=True)
    return root


def persist_input(
    storage_root: Path,
    source: str | None,
) -> str:
    if not source:
        return ""

    src = Path(source)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Fichier uploadé introuvable: {src}")

    store = ensure_input_store(storage_root)
    try:
        if src.parent.resolve() == store.resolve():
            return str(src)
    except Exception:
        pass

    digest = hash_file(src)
    suffix = src.suffix.lower()

    existing = next(
        (
            candidate
            for candidate in store.glob(f"{digest}.*")
            if candidate.is_file() and not candidate.name.startswith(".")
        ),
        None,
    )
    if existing is not None:
        return str(existing)

    dest = store / f"{digest}{suffix}"
    if not dest.exists() or dest.stat().st_size != src.stat().st_size:
        tmp = store / f".{digest}.{uuid.uuid4().hex[:8]}.part"
        try:
            shutil.copy2(src, tmp)
            if hash_file(tmp) != digest:
                raise RuntimeError("Checksum invalide après copie de l'upload.")
            os.replace(tmp, dest)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    return str(dest)


def unreferenced_store_files(
    storage_root: Path,
    *,
    referenced_paths: set[str],
) -> list[Path]:
    store = storage_root / "_input_store"
    if not store.exists():
        return []

    result: list[Path] = []
    for path in store.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = str(path)
        if resolved not in referenced_paths:
            result.append(path)
    return result

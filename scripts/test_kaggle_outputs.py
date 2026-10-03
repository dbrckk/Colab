import hashlib
import json
import tempfile
from pathlib import Path

from kaggle_app.kaggle_outputs import validate_downloaded_outputs


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def media_ok(path: Path, kind: str) -> bool:
    return path.is_file() and path.stat().st_size > 0 and kind in {"image", "video"}


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    image = root / "image.png"
    image.write_bytes(b"image-bytes")
    manifest = root / "result.json"
    manifest.write_text(
        json.dumps({
            "status": "done",
            "task": "image",
            "worker_version": "1.3",
            "files": ["image.png"],
            "output_manifest": [{
                "name": "image.png",
                "size": image.stat().st_size,
                "sha256": hash_file(image),
            }],
        }),
        encoding="utf-8",
    )
    data = validate_downloaded_outputs(
        {"task": "image", "meta": {}},
        [(manifest, "file"), (image, "image")],
        hash_file=hash_file,
        media_integrity_ok=media_ok,
    )
    assert data["status"] == "done"

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    image = root / "image.png"
    image.write_bytes(b"x")
    manifest = root / "result.json"
    manifest.write_text(
        json.dumps({
            "status": "done",
            "worker_version": "1.3",
            "files": ["image.png"],
        }),
        encoding="utf-8",
    )
    try:
        validate_downloaded_outputs(
            {"task": "image", "meta": {}},
            [(manifest, "file"), (image, "image")],
            hash_file=hash_file,
            media_integrity_ok=media_ok,
        )
        raise AssertionError("missing SHA-256 manifest accepted")
    except RuntimeError as exc:
        assert "Manifest SHA-256 absent" in str(exc)

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    images = []
    entries = []
    for i in range(2):
        p = root / f"image_{i}.png"
        p.write_bytes(f"img-{i}".encode())
        images.append((p, "image"))
        entries.append({
            "name": p.name,
            "size": p.stat().st_size,
            "sha256": hash_file(p),
        })
    manifest = root / "result.json"
    manifest.write_text(
        json.dumps({
            "status": "done",
            "worker_version": "1.3",
            "files": [p.name for p, _ in images],
            "output_manifest": entries,
            "batch_count": 2,
        }),
        encoding="utf-8",
    )
    try:
        validate_downloaded_outputs(
            {"task": "image_batch", "meta": {"batch_count": 3, "prompts": ["a", "b", "c"]}},
            [(manifest, "file"), *images],
            hash_file=hash_file,
            media_integrity_ok=media_ok,
        )
        raise AssertionError("incomplete batch accepted")
    except RuntimeError as exc:
        assert "Lot incomplet" in str(exc)

print("Kaggle output validation tests passed.")

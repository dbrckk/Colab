import json
import tempfile
import zipfile
from pathlib import Path

from kaggle_app.kaggle_export import build_job_archive
from kaggle_app.kaggle_input_store import hash_file


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    media = root / "result.png"
    media.write_bytes(b"image-bytes")

    archive = Path(build_job_archive(
        export_root=root / "exports",
        job_id="job123",
        job={"id": "job123", "task": "image", "status": "done"},
        artifacts=[{"id": 1, "path": str(media), "kind": "image"}],
        hash_file=hash_file,
        exported_at="2026-10-05 00:00:00",
    ))

    assert archive.is_file()
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        assert "job.json" in names
        assert "media/result.png" in names
        manifest = json.loads(zf.read("job.json").decode("utf-8"))
        assert manifest["job"]["id"] == "job123"
        assert manifest["exported_at"] == "2026-10-05 00:00:00"
        exported = manifest["exported_artifacts"]
        assert len(exported) == 1
        assert exported[0]["name"] == "result.png"
        assert exported[0]["size"] == len(b"image-bytes")
        assert len(exported[0]["sha256"]) == 64

print("Kaggle job archive builder tests passed.")

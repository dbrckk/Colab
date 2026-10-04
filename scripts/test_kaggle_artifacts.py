import tempfile
from pathlib import Path

from kaggle_app.kaggle_artifacts import stale_artifact_ids


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    present = root / "present.png"
    present.write_bytes(b"x")

    rows = [
        {"id": 1, "path": str(present)},
        {"id": 2, "path": str(root / "missing.png")},
        {"id": 3, "path": ""},
    ]
    assert stale_artifact_ids(rows) == [2, 3]

print("Kaggle stale artifact detection tests passed.")

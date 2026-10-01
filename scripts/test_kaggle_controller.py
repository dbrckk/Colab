from pathlib import Path
import tempfile

from kaggle_app.db import JobDB
from kaggle_app.kaggle_runner import slugify
from kaggle_app.storage import kind_for, import_outputs

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    db = JobDB(root / "jobs.sqlite3")
    db.create_job("j1", "image", "hello", {"steps": 20})
    row = db.get_job("j1")
    assert row and row["status"] == "queued"
    assert row["meta"]["steps"] == 20
    db.update_job("j1", status="done")
    assert db.get_job("j1")["status"] == "done"

    outputs = root / "outputs"
    outputs.mkdir()
    (outputs / "a.png").write_bytes(b"png-demo")
    (outputs / "b.mp4").write_bytes(b"mp4-demo")
    imported = import_outputs("j1", outputs, root / "storage")
    assert {kind for _, kind in imported} == {"image", "video"}

assert slugify("Qwen Studio JOB 123!") == "qwen-studio-job-123"
assert kind_for(Path("x.png")) == "image"
assert kind_for(Path("x.mp4")) == "video"
assert kind_for(Path("x.json")) == "file"

print("Kaggle controller tests passed.")


import json

repo_root = Path(__file__).resolve().parents[1]
controller_nb = repo_root / "Qwen_Kaggle_Studio_Controller.ipynb"
nb = json.loads(controller_nb.read_text(encoding="utf-8"))
assert nb["nbformat"] == 4
controller_code = "\n".join(
    "".join(cell.get("source", []))
    for cell in nb.get("cells", [])
    if cell.get("cell_type") == "code"
)
assert "drive.mount('/content/drive')" in controller_code
assert "QWEN_KAGGLE_STORAGE" in controller_code
assert "QWEN_KAGGLE_DB" in controller_code
assert "QWEN_KAGGLE_ENV_FILE" in controller_code
assert "launch_kaggle_ui.py" in controller_code
print("Kaggle controller notebook validation passed.")

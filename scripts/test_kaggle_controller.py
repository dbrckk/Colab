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


import os
from kaggle_app.config import Settings
from kaggle_app.kaggle_runner import KaggleController

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    os.environ["KAGGLE_USERNAME"] = "ci-user"
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=5,
        kernel_timeout=3600,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=True,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    folder = tmp / "kernel"
    folder.mkdir()
    job = {
        "id": "ci-job",
        "task": "image",
        "prompt": "test prompt",
        "meta": {"steps": 20, "cfg": 1.0, "seed": 1, "aspect": "1:1"},
    }
    ref = controller._prepare_kernel(job, folder, "ci-user/ci-dataset")
    assert ref == "ci-user/qwen-studio-ci-job"
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((folder / "job.ipynb").read_text(encoding="utf-8"))
    assert metadata["kernel_type"] == "notebook"
    assert metadata["code_file"] == "job.ipynb"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["dataset_sources"] == ["ci-user/ci-dataset"]
    assert notebook["nbformat"] == 4
    worker_code = "".join(notebook["cells"][1]["source"])
    assert "def run_image()" in worker_code
    assert "def run_video_faceswap()" in worker_code
    controller.executor.shutdown(wait=False)

print("Generated Kaggle notebook validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=5,
        kernel_timeout=3600,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=True,
        keep_job_inputs=False,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    uploaded = tmp / "temporary-upload.jpg"
    uploaded.write_bytes(b"temporary-gradio-upload")
    persisted = Path(controller._persist_input("job-upload", str(uploaded), "source_image"))
    assert persisted.exists()
    assert persisted.read_bytes() == uploaded.read_bytes()
    uploaded.unlink()
    assert persisted.exists(), "persisted upload disappeared with temporary source"
    controller._cleanup_inputs("job-upload")
    assert not persisted.exists()
    controller.executor.shutdown(wait=False)

print("Persisted upload validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=5,
        kernel_timeout=3600,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=True,
        keep_job_inputs=False,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.db.create_job("cancel-me", "image", "x", {})
    # Simulate cancellation before a remote kernel exists.
    msg = controller.cancel("cancel-me")
    assert "Annulation demandée" in msg or "annulé" in msg.lower()
    assert controller.db.get_job("cancel-me")["status"] in {"cancel_requested", "cancelled"}
    controller.executor.shutdown(wait=False)

print("Queued cancellation validation passed.")

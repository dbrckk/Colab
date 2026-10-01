import base64
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
    assert "def run_image_edit()" in worker_code
    assert "def run_video_faceswap()" in worker_code
    assert "--llm_vision" in worker_code
    assert "--negative-prompt" in worker_code
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


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    source = root / "source"
    source.mkdir()
    storage = root / "storage"
    (source / "same.png").write_bytes(b"same-content")
    first = import_outputs("job-dedupe", source, storage)
    second = import_outputs("job-dedupe", source, storage)
    assert len(first) == 1 and len(second) == 1
    assert first[0][0] == second[0][0]
    assert len(list((storage / "job-dedupe").glob("*"))) == 1

print("Artifact deduplication validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1,
        kernel_timeout=60,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False,
        keep_job_inputs=False,
        share_gradio=False,
    )
    first = KaggleController(settings)
    first.db.create_job("stale-prep", "image", "x", {})
    first.db.update_job("stale-prep", status="preparing")
    first.db.create_job("remote-running", "image", "x", {})
    first.db.update_job("remote-running", status="running", kernel_ref="ci-user/kernel")
    first.executor.shutdown(wait=False)

    original_recover = KaggleController._recover_remote_job
    try:
        KaggleController._recover_remote_job = (
            lambda self, job_id: self.db.update_job(job_id, status="recovered-test")
        )
        second = KaggleController(settings)
        second.executor.shutdown(wait=True)
        assert second.db.get_job("stale-prep")["status"] == "interrupted"
        assert second.db.get_job("remote-running")["status"] == "recovered-test"
    finally:
        KaggleController._recover_remote_job = original_recover

print("Controller restart recovery validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1,
        kernel_timeout=60,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False,
        keep_job_inputs=False,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.db.create_job("retry-old", "image", "hello", {"steps": 12, "cfg": 1.0, "seed": 2, "aspect": "1:1"})
    controller.db.update_job("retry-old", status="error", error="boom")

    # Replace submit so retry can be tested without Kaggle credentials/threads.
    submitted = {}
    def fake_submit(task, prompt, negative_prompt="", steps=25, cfg=1.0, seed=-1, aspect="1:1", source_image=None, target_video=None):
        submitted.update({
            "task": task, "prompt": prompt, "steps": steps,
            "cfg": cfg, "seed": seed, "aspect": aspect,
        })
        return "retry-new"
    controller.submit = fake_submit
    assert controller.retry("retry-old") == "retry-new"
    assert submitted["task"] == "image"
    assert submitted["prompt"] == "hello"
    assert submitted["steps"] == 12

    media_dir = settings.storage_root / "delete-me"
    media_dir.mkdir(parents=True)
    (media_dir / "x.png").write_bytes(b"x")
    controller.db.create_job("delete-me", "image", "x", {})
    controller.db.update_job("delete-me", status="done")
    assert "supprimé" in controller.delete_local_job("delete-me")
    assert controller.db.get_job("delete-me") is None
    assert not media_dir.exists()
    controller.executor.shutdown(wait=False)

print("Retry and delete validation passed.")


worker_path = repo_root / "kaggle_worker" / "worker.py"
worker_source = worker_path.read_text(encoding="utf-8")
compile(worker_source, str(worker_path), "exec")
for token in [
    "def run_image()",
    "def run_image_edit()",
    "def run_video_faceswap()",
    "def load_qwen_models(",
    "WORKER_VERSION = \"1.2\"",
    "--negative-prompt",
    "--llm_vision",
]:
    assert token in worker_source, f"Missing worker capability: {token}"
print("Kaggle worker syntax validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1,
        kernel_timeout=60,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False,
        keep_job_inputs=False,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.db.create_job("q1", "image", "a", {})
    controller.db.create_job("q2", "image", "b", {})
    controller.db.update_job("q1", status="queued")
    controller.db.update_job("q2", status="queued")
    assert controller.queue_position("q1") == 1
    assert controller.queue_position("q2") == 2

    media = settings.storage_root / "q1"
    media.mkdir(parents=True)
    (media / "a.bin").write_bytes(b"x" * 1234)
    assert controller.job_storage_bytes("q1") == 1234
    controller.executor.shutdown(wait=False)

print("Queue and storage validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1,
        kernel_timeout=60,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False,
        keep_job_inputs=False,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    uploaded = tmp / "source.png"
    uploaded.write_bytes(b"source")
    persisted = Path(controller._persist_input("cancel-input", str(uploaded), "source_image"))
    controller.db.create_job(
        "cancel-input",
        "image_edit",
        "edit",
        {"source_image": str(persisted)},
    )
    controller.db.update_job("cancel-input", status="queued")
    controller.cancel("cancel-input")
    assert persisted.exists(), "cancel unexpectedly removed retryable input"
    controller.executor.shutdown(wait=False)

print("Cancelled input retention validation passed.")


from kaggle_app.kaggle_runner import _is_transient_cli_error

assert _is_transient_cli_error("503 Service Unavailable")
assert _is_transient_cli_error("connection reset by peer")
assert _is_transient_cli_error("429 Too Many Requests")
assert not _is_transient_cli_error("401 Unauthorized")
assert not _is_transient_cli_error("usage: kaggle kernels push")
print("Transient Kaggle error classification passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1,
        cli_retries=1,
        kernel_timeout=60,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False,
        keep_job_inputs=False,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    job = {"id": "ref-test", "task": "image", "prompt": "x", "meta": {}}
    assert controller._dataset_ref(job) == "ci-user/qwen-input-ref-test"
    controller.executor.shutdown(wait=False)

print("Remote dataset reference persistence validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1,
        cli_retries=1,
        kernel_timeout=60,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False,
        keep_job_inputs=False,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    submitted = []
    def fake_submit(task, prompt, negative_prompt="", steps=25, cfg=1.0, seed=-1, aspect="1:1", source_image=None, target_video=None):
        submitted.append((task, prompt, seed))
        return f"id-{len(submitted)}"
    controller.submit = fake_submit
    ids = controller.submit_batch("first\n\nsecond\nthird", seed=100)
    assert ids == ["id-1", "id-2", "id-3"]
    assert submitted == [
        ("image", "first", 100),
        ("image", "second", 101),
        ("image", "third", 102),
    ]
    try:
        controller.submit_batch("\n".join(f"p{i}" for i in range(21)))
        raise AssertionError("batch limit was not enforced")
    except ValueError:
        pass
    controller.executor.shutdown(wait=False)

print("Batch submission validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    db = JobDB(tmp / "jobs.sqlite3")
    db.create_job("lib1", "image", "prompt one", {})
    db.update_job("lib1", status="done")
    db.add_artifact("lib1", str(tmp / "one.png"), "image")
    db.add_artifact("lib1", str(tmp / "one.json"), "file")
    recent = db.recent_artifacts("image", 10)
    assert len(recent) == 1
    assert recent[0]["job_id"] == "lib1"
    assert recent[0]["prompt"] == "prompt one"
    assert recent[0]["kind"] == "image"

print("Recent artifact library validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1,
        cli_retries=1,
        kernel_timeout=60,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False,
        keep_job_inputs=False,
        keep_source_inputs_for_retry=True,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    uploaded = tmp / "video.mp4"
    uploaded.write_bytes(b"video")
    persisted = Path(controller._persist_input("retry-source", str(uploaded), "target_video"))
    controller.db.create_job(
        "retry-source",
        "video_faceswap",
        "",
        {"target_video": str(persisted), "source_image": str(persisted)},
    )
    controller.db.update_job("retry-source", status="done")
    controller._cleanup_inputs("retry-source")
    assert persisted.exists(), "source-backed completed job lost retry input"
    controller._cleanup_inputs("retry-source", force=True)
    assert not persisted.exists()
    controller.executor.shutdown(wait=False)

print("Source-backed retry retention validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1,
        cli_retries=1,
        kernel_timeout=60,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False,
        keep_job_inputs=False,
        keep_source_inputs_for_retry=True,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.db.create_job("zip-job", "image", "zip prompt", {"steps": 15})
    controller.db.update_job("zip-job", status="done")
    media = settings.storage_root / "zip-job"
    media.mkdir(parents=True)
    image_path = media / "image.png"
    image_path.write_bytes(b"image")
    controller.db.add_artifact("zip-job", str(image_path), "image")
    archive = Path(controller.export_job_archive("zip-job"))
    assert archive.exists()
    import zipfile
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        assert "job.json" in names
        assert "media/image.png" in names
    controller.executor.shutdown(wait=False)

print("Job ZIP export validation passed.")


# Verify safe commands retry transient failures while permanent auth failures stop immediately.
import kaggle_app.kaggle_runner as runner_mod

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1,
        cli_retries=3,
        kernel_timeout=60,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False,
        keep_job_inputs=False,
        keep_source_inputs_for_retry=True,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.ensure_cli = lambda: "kaggle"
    original_run = runner_mod.subprocess.run
    original_sleep = runner_mod.time.sleep
    calls = []
    class FakeProc:
        def __init__(self, code, text):
            self.returncode = code
            self.stdout = ""
            self.stderr = text
    try:
        seq = [FakeProc(1, "503 Service Unavailable"), FakeProc(0, "ok")]
        def fake_run(*args, **kwargs):
            calls.append(args)
            return seq.pop(0)
        runner_mod.subprocess.run = fake_run
        runner_mod.time.sleep = lambda *_: None
        assert controller._run(["kernels", "status", "x"], retries=3) == "ok"
        assert len(calls) == 2

        calls.clear()
        runner_mod.subprocess.run = lambda *args, **kwargs: (
            calls.append(args) or FakeProc(1, "401 Unauthorized")
        )
        try:
            controller._run(["kernels", "status", "x"], retries=3)
            raise AssertionError("permanent auth failure was retried/accepted")
        except RuntimeError:
            pass
        assert len(calls) == 1
    finally:
        runner_mod.subprocess.run = original_run
        runner_mod.time.sleep = original_sleep
        controller.executor.shutdown(wait=False)

print("Kaggle CLI retry execution validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1,
        cli_retries=1,
        kernel_timeout=60,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False,
        keep_job_inputs=False,
        keep_source_inputs_for_retry=True,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    result_json = tmp / "result.json"
    image = tmp / "image.png"
    video = tmp / "video.mp4"

    result_json.write_text('{"status":"done"}', encoding="utf-8")
    image.write_bytes(b"image-data")
    video.write_bytes(b"video-data")
    controller._media_integrity_ok = lambda path, kind: path.exists() and path.stat().st_size > 0

    manifest = controller._validate_downloaded_outputs(
        {"task": "image"},
        [(result_json, "file"), (image, "image")],
    )
    assert manifest["status"] == "done"
    controller._validate_downloaded_outputs(
        {"task": "video_faceswap"},
        [(result_json, "file"), (video, "video")],
    )

    try:
        controller._validate_downloaded_outputs(
            {"task": "image"},
            [(result_json, "file")],
        )
        raise AssertionError("image job accepted without image output")
    except RuntimeError:
        pass

    result_json.write_text('{"status":"error","error":"boom"}', encoding="utf-8")
    try:
        controller._validate_downloaded_outputs(
            {"task": "image"},
            [(result_json, "file"), (image, "image")],
        )
        raise AssertionError("error manifest was accepted")
    except RuntimeError:
        pass

    controller.executor.shutdown(wait=False)

print("Downloaded output validation passed.")


# Full controller state-machine integration test without contacting Kaggle.
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1,
        cli_retries=1,
        kernel_timeout=60,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False,
        keep_job_inputs=False,
        keep_source_inputs_for_retry=True,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.db.create_job(
        "e2e-job",
        "image",
        "integration prompt",
        {
            "negative_prompt": "",
            "steps": 12,
            "cfg": 1.0,
            "seed": 42,
            "aspect": "1:1",
            "source_image": "",
            "target_video": "",
        },
    )

    controller._prepare_dataset = lambda job, folder, dataset_ref=None: (
        dataset_ref or "ci-user/qwen-input-e2e-job"
    )
    controller._kernel_status = lambda ref: ("complete", "complete")

    def fake_run(args, timeout=None, retries=None):
        if args[:2] == ["kernels", "output"]:
            out_dir = Path(args[args.index("-p") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "result.json").write_text(
                json.dumps({
                    "status": "done",
                    "task": "image",
                    "worker_version": "ci",
                    "seed": 42,
                    "width": 1024,
                    "height": 1024,
                    "steps": 12,
                    "files": ["image.png"],
                }),
                encoding="utf-8",
            )
            (out_dir / "image.png").write_bytes(base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
            ))
        return "ok"

    controller._run = fake_run
    controller._execute("e2e-job")

    finished = controller.db.get_job("e2e-job")
    assert finished["status"] == "done", finished
    assert finished["meta"]["result_manifest"]["seed"] == 42
    artifacts = controller.db.artifacts("e2e-job")
    assert any(a["kind"] == "image" for a in artifacts)
    assert any(Path(a["path"]).name == "result.json" for a in artifacts)
    controller.executor.shutdown(wait=False)

print("Simulated end-to-end Kaggle controller flow passed.")


# Real image integrity check with a minimal valid PNG.
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1,
        cli_retries=1,
        kernel_timeout=60,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False,
        keep_job_inputs=False,
        keep_source_inputs_for_retry=True,
        share_gradio=False,
    )
    controller = KaggleController(settings)
    valid_png = tmp / "valid.png"
    valid_png.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    ))
    invalid_png = tmp / "invalid.png"
    invalid_png.write_bytes(b"not-an-image")
    assert controller._media_integrity_ok(valid_png, "image")
    assert not controller._media_integrity_ok(invalid_png, "image")
    controller.executor.shutdown(wait=False)

print("Media integrity validation passed.")


# Local queued jobs survive controller restart and wait for auth when needed.
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1,
        cli_retries=1,
        kernel_timeout=60,
        accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False,
        keep_job_inputs=False,
        keep_source_inputs_for_retry=True,
        share_gradio=False,
    )

    old_user = os.environ.pop("KAGGLE_USERNAME", None)
    old_token = os.environ.pop("KAGGLE_API_TOKEN", None)
    old_key = os.environ.pop("KAGGLE_KEY", None)
    try:
        first = KaggleController(settings)
        first.db.create_job("local-queued", "image", "hello", {})
        first.db.update_job("local-queued", status="queued")
        first.executor.shutdown(wait=False)

        second = KaggleController(settings)
        assert second.db.get_job("local-queued")["status"] == "waiting_auth"
        second.executor.shutdown(wait=False)
    finally:
        if old_user is not None:
            os.environ["KAGGLE_USERNAME"] = old_user
        if old_token is not None:
            os.environ["KAGGLE_API_TOKEN"] = old_token
        if old_key is not None:
            os.environ["KAGGLE_KEY"] = old_key

print("Local queued-job restart validation passed.")

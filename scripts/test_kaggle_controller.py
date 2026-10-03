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
assert "userdata.get" in controller_code
assert "KAGGLE_API_TOKEN" in controller_code
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
    controller.db.create_job(
        "job-upload",
        "image_edit",
        "edit",
        {"source_image": str(persisted)},
    )
    controller._cleanup_inputs("job-upload")
    assert persisted.exists(), "shared canonical input was deleted by ordinary cleanup"
    controller.db.delete_job("job-upload")
    removed, _ = controller._gc_input_store()
    assert removed == 1
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
    "def run_image_batch()",
    "batch_backend=batch_backend",
    "/sdapi/v1/txt2img",
    "def _server_txt2img(",
    "def ensure_sdserver(",
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
        keep_source_inputs_for_retry=True,
        share_gradio=False,
    )
    os.environ["KAGGLE_USERNAME"] = "ci-user"
    os.environ["KAGGLE_API_TOKEN"] = "ci-token"
    controller = KaggleController(settings)
    original_execute = controller._execute
    controller._execute = lambda job_id: None
    try:
        ids = controller.submit_batch("first\n\nsecond\nthird", seed=100)
        assert len(ids) == 1
        batch_job = controller.db.get_job(ids[0])
        assert batch_job["task"] == "image_batch"
        assert batch_job["meta"]["prompts"] == ["first", "second", "third"]
        assert batch_job["meta"]["batch_count"] == 3
        assert batch_job["meta"]["seed"] == 100
        try:
            controller.submit_batch("\n".join(f"p{i}" for i in range(21)))
            raise AssertionError("batch limit was not enforced")
        except ValueError:
            pass
    finally:
        controller._execute = original_execute
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
    controller.delete_local_job("retry-source")
    assert not persisted.exists(), "unreferenced canonical input was not garbage-collected"
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


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    corrupt = tmp / "corrupt.sqlite3"
    corrupt.write_bytes(b"this is not a sqlite database")
    db = JobDB(corrupt)
    assert db.integrity_status().lower().startswith("ok")
    assert db.recovered_corrupt_path is not None
    assert db.recovered_corrupt_path.exists()
    db.create_job("after-recovery", "image", "ok", {})
    assert db.get_job("after-recovery") is not None

print("SQLite corruption recovery validation passed.")


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
    controller.db.create_job("known", "image", "x", {})

    orphan = settings.storage_root / "_inputs" / "orphan"
    orphan.mkdir(parents=True)
    (orphan / "x.bin").write_bytes(b"x" * 100)

    known = settings.storage_root / "_inputs" / "known"
    known.mkdir(parents=True)
    (known / "keep.bin").write_bytes(b"keep")

    exports = settings.storage_root / "_exports"
    exports.mkdir(parents=True)
    old_zip = exports / "old.zip"
    old_zip.write_bytes(b"z" * 100)
    import os as _os
    old_time = __import__("time").time() - 20 * 86400
    _os.utime(old_zip, (old_time, old_time))

    msg = controller.cleanup_storage(export_max_age_days=14)
    assert "Nettoyage terminé" in msg
    assert not orphan.exists()
    assert known.exists()
    assert not old_zip.exists()
    controller.executor.shutdown(wait=False)

print("Safe storage cleanup validation passed.")


from kaggle_app.storage import scan_outputs

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    download = tmp / "download"
    download.mkdir()
    (download / "result.json").write_text('{"status":"done","files":["image.png"]}', encoding="utf-8")
    (download / "image.png").write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    ))
    staged = scan_outputs(download)
    assert {kind for _, kind in staged} == {"file", "image"}

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
    controller._validate_downloaded_outputs({"task":"image"}, staged)
    assert not settings.storage_root.joinpath("precheck").exists()
    controller.executor.shutdown(wait=False)

print("Pre-persistence validation flow passed.")


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
    plain_job = {
        "id": "plain-image",
        "task": "image",
        "prompt": "hello inline",
        "meta": {"steps": 21, "cfg": 1.0, "seed": 7, "aspect": "1:1"},
    }
    assert not controller._needs_dataset(plain_job)
    kernel_dir = tmp / "inline-kernel"
    kernel_dir.mkdir()
    ref = controller._prepare_kernel(plain_job, kernel_dir, "")
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((kernel_dir / "job.ipynb").read_text(encoding="utf-8"))
    assert metadata["dataset_sources"] == []
    assert ref == "ci-user/qwen-studio-plain-image"
    assert len(notebook["cells"]) == 3
    bootstrap = "".join(notebook["cells"][1]["source"])
    assert "/kaggle/working/job_config.json" in bootstrap
    assert "hello inline" in bootstrap
    worker = "".join(notebook["cells"][2]["source"])
    assert "def run_image()" in worker

    source_job = {
        "id": "edit-image",
        "task": "image_edit",
        "prompt": "edit",
        "meta": {"source_image": "/tmp/source.png"},
    }
    assert controller._needs_dataset(source_job)
    controller.executor.shutdown(wait=False)

print("Source-free inline Kaggle config validation passed.")


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
    job = {
        "id": "batch-one-kernel",
        "task": "image_batch",
        "prompt": "Lot de 2 images",
        "meta": {
            "prompts": ["one", "two"],
            "batch_count": 2,
            "steps": 20,
            "cfg": 1.0,
            "seed": 10,
            "aspect": "1:1",
        },
    }
    assert not controller._needs_dataset(job)
    folder = tmp / "kernel"
    folder.mkdir()
    controller._prepare_kernel(job, folder, "")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((folder / "job.ipynb").read_text(encoding="utf-8"))
    assert metadata["dataset_sources"] == []
    bootstrap = "".join(notebook["cells"][1]["source"])
    assert '"prompts": ["one", "two"]' in bootstrap
    worker = "".join(notebook["cells"][2]["source"])
    assert "def run_image_batch()" in worker
    controller.executor.shutdown(wait=False)

print("Single-kernel batch notebook validation passed.")


# Source-free image execution must never create/upload a Kaggle dataset.
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
        "no-dataset-execute",
        "image",
        "plain prompt",
        {
            "negative_prompt": "",
            "steps": 10,
            "cfg": 1.0,
            "seed": 1,
            "aspect": "1:1",
            "source_image": "",
            "target_video": "",
        },
    )

    def dataset_must_not_run(*args, **kwargs):
        raise AssertionError("_prepare_dataset was called for a source-free image job")

    controller._prepare_dataset = dataset_must_not_run
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
                    "seed": 1,
                    "files": ["image.png"],
                }),
                encoding="utf-8",
            )
            (out_dir / "image.png").write_bytes(base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
            ))
        return "ok"

    controller._run = fake_run
    controller._execute("no-dataset-execute")
    finished = controller.db.get_job("no-dataset-execute")
    assert finished["status"] == "done", finished
    assert (finished["meta"].get("dataset_ref") or "") == ""
    controller.executor.shutdown(wait=False)

print("Source-free execute skips dataset validation passed.")


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
    upload1 = tmp / "video-a.mp4"
    upload2 = tmp / "video-b.mp4"
    payload = b"same-large-video-content" * 100
    upload1.write_bytes(payload)
    upload2.write_bytes(payload)

    p1 = Path(controller._persist_input("shared-1", str(upload1), "target_video"))
    p2 = Path(controller._persist_input("shared-2", str(upload2), "target_video"))
    assert p1 == p2
    assert p1.parent.name == "_input_store"
    assert len(list(p1.parent.iterdir())) == 1

    controller.db.create_job("shared-1", "video_faceswap", "", {"target_video": str(p1)})
    controller.db.create_job("shared-2", "video_faceswap", "", {"target_video": str(p2)})
    controller.db.update_job("shared-1", status="done")
    controller.db.update_job("shared-2", status="done")

    controller.delete_local_job("shared-1")
    assert p1.exists(), "shared source deleted while another job still references it"
    controller.delete_local_job("shared-2")
    assert not p1.exists(), "last shared source reference did not trigger GC"
    controller.executor.shutdown(wait=False)

print("Content-addressed input deduplication validation passed.")


# Batch jobs should prefer a persistent sd-server model context and retain CLI fallback.
assert "ensure_sdserver(sdcli)" in worker_source
assert '"/sdapi/v1/txt2img"' in worker_source
assert '"--conditioning-cache-size", "4"' in worker_source
assert 'batch_backend = "sd-server"' in worker_source
assert 'batch_backend = "sd-cli"' in worker_source
assert "server_process.terminate()" in worker_source
print("Batch server API wiring validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    db = JobDB(tmp / "jobs.sqlite3")
    db.create_job("meta-list", "image_edit", "x", {"source_image": "/tmp/example.png", "steps": 33})
    rows = db.list_jobs(10)
    row = next(j for j in rows if j["id"] == "meta-list")
    assert row["meta"]["source_image"] == "/tmp/example.png"
    assert row["meta"]["steps"] == 33

print("list_jobs metadata decoding validation passed.")


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
    a = tmp / "same.mp4"
    b = tmp / "same.mov"
    payload = b"same-bytes-with-different-extension"
    a.write_bytes(payload)
    b.write_bytes(payload)
    pa = Path(controller._persist_input("x1", str(a), "target_video"))
    pb = Path(controller._persist_input("x2", str(b), "target_video"))
    assert pa == pb
    assert len([p for p in pa.parent.iterdir() if p.is_file() and not p.name.startswith(".")]) == 1
    controller.executor.shutdown(wait=False)

print("Cross-extension input deduplication validation passed.")


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
    controller.db.create_job("wait-cancel", "image", "x", {})
    controller.db.update_job("wait-cancel", status="waiting_auth")
    msg = controller.cancel("wait-cancel")
    assert "annulé" in msg.lower()
    assert controller.db.get_job("wait-cancel")["status"] == "cancelled"
    controller.executor.shutdown(wait=False)

print("Waiting-auth cancellation validation passed.")


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
    good = tmp / "good.png"
    good.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    ))
    bad = tmp / "bad.png"
    bad.write_bytes(b"broken")

    controller._validate_submission("image", "ok", 25, 1.0, -1, "1:1")
    controller._validate_submission("image_edit", "edit", 25, 1.0, 1, "16:9", str(good), None)

    invalid_cases = [
        ("image", "", 25, 1.0, -1, "1:1", None, None),
        ("image", "x", 0, 1.0, -1, "1:1", None, None),
        ("image", "x", 25, 31.0, -1, "1:1", None, None),
        ("image", "x", 25, 1.0, -2, "1:1", None, None),
        ("image", "x", 25, 1.0, -1, "2:1", None, None),
        ("image_edit", "x", 25, 1.0, -1, "1:1", str(bad), None),
    ]
    for args in invalid_cases:
        try:
            controller._validate_submission(*args)
            raise AssertionError(f"invalid submission accepted: {args}")
        except (ValueError, FileNotFoundError):
            pass
    controller.executor.shutdown(wait=False)

print("Submission preflight validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root,
        storage_root=tmp / "media",
        db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local",
        worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60,
        accelerator="NvidiaTeslaT4", delete_remote_kernel=False,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    result = tmp / "result.json"
    image = tmp / "image_001.png"
    image.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    ))
    result.write_text(json.dumps({
        "status": "done", "batch_count": 1, "files": ["image_001.png"]
    }), encoding="utf-8")
    job = {"task": "image_batch", "meta": {"batch_count": 2, "prompts": ["a", "b"]}}
    try:
        controller._validate_downloaded_outputs(job, [(result, "file"), (image, "image")])
        raise AssertionError("incomplete batch accepted")
    except RuntimeError as exc:
        assert "Lot incomplet" in str(exc)
    controller.executor.shutdown(wait=False)

print("Incomplete batch validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, keep_job_inputs=False,
        keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.db.create_job("recover-only", "image", "x", {})
    controller.db.update_job(
        "recover-only", status="error", kernel_ref="ci-user/kernel",
        meta_json={"recover_outputs_available": True, "failed_phase": "downloading"},
    )
    called = []
    controller._recover_remote_job = lambda job_id: called.append(job_id)
    msg = controller.recover_outputs("recover-only")
    assert "sans recalcul GPU" in msg
    controller.executor.shutdown(wait=True)
    assert called == ["recover-only"]
    assert controller.db.get_job("recover-only")["status"] == "recovering"

print("Output-only recovery scheduling passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=True, keep_job_inputs=False,
        keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.db.create_job("recover-repeat", "image", "x", {})
    controller.db.update_job(
        "recover-repeat", status="error", kernel_ref="ci-user/completed-kernel",
        meta_json={"recover_outputs_available": True, "failed_phase": "downloading"},
    )
    controller._kernel_status = lambda ref: ("complete", "complete")
    cleanup_calls = []
    controller._cleanup_remote_refs = lambda kernel_ref="", dataset_ref="", force=False: cleanup_calls.append(
        (kernel_ref, dataset_ref, force)
    )
    controller._run = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("temporary download failure"))
    controller._recover_remote_job("recover-repeat")
    recovered = controller.db.get_job("recover-repeat")
    assert recovered["status"] == "error"
    assert recovered["meta"]["recover_outputs_available"] is True
    assert recovered["kernel_ref"] == "ci-user/completed-kernel"
    assert not any(call[0] == "ci-user/completed-kernel" for call in cleanup_calls)
    controller.executor.shutdown(wait=False)

print("Repeated output recovery retention passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, keep_job_inputs=False,
        keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.db.create_job("no-recovery-source", "image", "x", {})
    controller.db.update_job(
        "no-recovery-source", status="error", kernel_ref="ci-user/old-kernel",
        meta_json={"recover_outputs_available": False},
    )
    try:
        controller.recover_outputs("no-recovery-source")
        raise AssertionError("recovery accepted without a preserved output source")
    except ValueError as exc:
        assert "Relancer" in str(exc)
    controller.executor.shutdown(wait=False)

print("Recovery availability guard passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    db_path = tmp / "jobs.sqlite3"
    db = JobDB(db_path)
    db.create_job("artifact-idempotent", "image", "x", {})
    artifact_path = str(tmp / "same.png")
    db.add_artifact("artifact-idempotent", artifact_path, "image")
    db.add_artifact("artifact-idempotent", artifact_path, "image")
    rows = db.artifacts("artifact-idempotent")
    assert len(rows) == 1
    assert rows[0]["path"] == artifact_path

print("Artifact DB idempotency validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    db_path = tmp / "legacy.sqlite3"
    import sqlite3 as _sqlite3
    con = _sqlite3.connect(db_path)
    con.executescript("""
    CREATE TABLE jobs (
        id TEXT PRIMARY KEY, task TEXT NOT NULL, prompt TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL, kernel_ref TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL, updated_at REAL NOT NULL,
        error TEXT NOT NULL DEFAULT '', meta_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE artifacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
        path TEXT NOT NULL, kind TEXT NOT NULL, created_at REAL NOT NULL
    );
    INSERT INTO jobs VALUES ('legacy','image','x','done','',1,1,'','{}');
    INSERT INTO artifacts(job_id,path,kind,created_at) VALUES ('legacy','/tmp/a.png','image',1);
    INSERT INTO artifacts(job_id,path,kind,created_at) VALUES ('legacy','/tmp/a.png','image',2);
    """)
    con.commit()
    con.close()
    db = JobDB(db_path)
    assert len(db.artifacts("legacy")) == 1
    db.add_artifact("legacy", "/tmp/a.png", "image")
    assert len(db.artifacts("legacy")) == 1

print("Legacy artifact duplicate migration passed.")


from kaggle_app.storage import import_outputs as _import_outputs

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    src = tmp / "download"
    src.mkdir()
    (src / "image.png").write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    ))
    storage = tmp / "media"
    first = _import_outputs("atomic", src, storage)
    second = _import_outputs("atomic", src, storage)
    assert first[0][0] == second[0][0]
    assert len([p for p in (storage / "atomic").iterdir() if p.is_file()]) == 1
    assert not list((storage / "atomic").glob("*.part"))
    assert not list((storage / "atomic").glob(".*.part"))

print("Atomic output import validation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, keep_job_inputs=False,
        keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.db.create_job("stale-artifact", "image", "x", {})
    missing = tmp / "media" / "stale-artifact" / "gone.png"
    controller.db.add_artifact("stale-artifact", str(missing), "image")
    checked, removed = controller.reconcile_artifacts()
    assert checked >= 1
    assert removed == 1
    assert controller.db.artifacts("stale-artifact") == []
    controller.executor.shutdown(wait=False)

print("Artifact reconciliation validation passed.")

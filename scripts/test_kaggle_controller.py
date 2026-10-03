import base64
import time
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
    original_execute = KaggleController._execute
    recovered = []
    executed = []
    old_values = {key: os.environ.pop(key, None) for key in (
        "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY"
    )}
    try:
        KaggleController._recover_remote_job = lambda self, job_id: recovered.append(job_id)
        KaggleController._execute = lambda self, job_id: executed.append(job_id)
        second = KaggleController(settings)
        second.executor.shutdown(wait=True)
        assert second.db.get_job("stale-prep")["status"] == "waiting_auth"
        assert second.db.get_job("remote-running")["status"] == "waiting_auth"
        assert second.db.get_job("remote-running")["kernel_ref"] == "ci-user/kernel"
        assert recovered == []
        assert executed == []
    finally:
        KaggleController._recover_remote_job = original_recover
        KaggleController._execute = original_execute
        for key, value in old_values.items():
            if value is not None:
                os.environ[key] = value

print("Controller restart waits safely for authentication passed.")


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
    "--negative-prompt",
    "--llm_vision",
]:
    assert token in worker_source, f"Missing worker capability: {token}"
version_line = next(
    (line for line in worker_source.splitlines() if line.startswith("WORKER_VERSION = ")),
    "",
)
assert version_line, "Missing WORKER_VERSION"
version_text = version_line.split("=", 1)[1].strip().strip('"')
version_tuple = tuple(int(x) for x in version_text.split("."))
assert version_tuple >= (1, 3), f"Worker version too old: {version_text}"
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
    original_validate_credentials = controller.validate_current_credentials
    controller._execute = lambda job_id: None
    controller.validate_current_credentials = lambda max_age_seconds=300: None
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
        controller.validate_current_credentials = original_validate_credentials
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


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, keep_job_inputs=False,
        keep_source_inputs_for_retry=True, share_gradio=False,
    )
    seed_db = JobDB(settings.db_path)
    seed_db.create_job("resume-preparing", "image", "resume me", {
        "steps": 20, "cfg": 1.0, "seed": 1, "aspect": "1:1",
        "source_image": "", "target_video": "", "dataset_ref": "ci-user/stale-dataset",
    })
    seed_db.update_job("resume-preparing", status="preparing")

    original_execute = KaggleController._execute
    original_cleanup = KaggleController._cleanup_remote_refs
    old_user = os.environ.get("KAGGLE_USERNAME")
    old_token = os.environ.get("KAGGLE_API_TOKEN")
    executed = []
    cleaned = []
    KaggleController._execute = lambda self, job_id: executed.append(job_id)
    KaggleController._cleanup_remote_refs = lambda self, kernel_ref="", dataset_ref="", force=False: cleaned.append(
        (kernel_ref, dataset_ref, force)
    )
    try:
        os.environ["KAGGLE_USERNAME"] = "ci-user"
        os.environ["KAGGLE_API_TOKEN"] = "ci-token"
        controller = KaggleController(settings)
        controller.executor.shutdown(wait=True)
        recovered = controller.db.get_job("resume-preparing")
        assert recovered["status"] == "queued"
        assert recovered["meta"]["dataset_ref"] == ""
        assert executed == ["resume-preparing"]
        assert any(x[1] == "ci-user/stale-dataset" for x in cleaned)
    finally:
        KaggleController._execute = original_execute
        KaggleController._cleanup_remote_refs = original_cleanup
        if old_user is None:
            os.environ.pop("KAGGLE_USERNAME", None)
        else:
            os.environ["KAGGLE_USERNAME"] = old_user
        if old_token is None:
            os.environ.pop("KAGGLE_API_TOKEN", None)
        else:
            os.environ["KAGGLE_API_TOKEN"] = old_token

print("Persisted local preparation auto-resume passed.")


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
    controller.db.create_job("lost-done", "image", "x", {})
    controller.db.update_job("lost-done", status="done", kernel_ref="ci-user/still-complete")
    controller._kernel_status = lambda ref: ("complete", "complete")
    checked, recoverable = controller.reconcile_completed_jobs()
    job = controller.db.get_job("lost-done")
    assert checked == 1
    assert recoverable == 1
    assert job["status"] == "error"
    assert job["meta"]["recover_outputs_available"] is True
    assert job["meta"]["failed_phase"] == "local_artifacts_missing"
    controller.executor.shutdown(wait=False)

print("Lost completed output recovery detection passed.")


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
    controller.db.create_job("lost-unknown", "image", "x", {})
    controller.db.update_job("lost-unknown", status="done", kernel_ref="ci-user/gone")
    controller._kernel_status = lambda ref: (_ for _ in ()).throw(RuntimeError("404"))
    checked, recoverable = controller.reconcile_completed_jobs()
    job = controller.db.get_job("lost-unknown")
    assert checked == 1
    assert recoverable == 0
    assert job["status"] == "error"
    assert job["meta"].get("recover_outputs_available") is not True
    controller.executor.shutdown(wait=False)

print("Unconfirmed lost-output state validation passed.")


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
    image = tmp / "image.png"
    image.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    ))
    import hashlib as _hashlib
    digest = _hashlib.sha256(image.read_bytes()).hexdigest()
    result = tmp / "result.json"
    result.write_text(json.dumps({
        "status": "done", "files": ["image.png"],
        "output_manifest": [{"name": "image.png", "size": image.stat().st_size, "sha256": digest}],
    }), encoding="utf-8")
    job = {"task": "image", "meta": {}}
    parsed = controller._validate_downloaded_outputs(job, [(result, "file"), (image, "image")])
    assert parsed["output_manifest"][0]["sha256"] == digest
    bad = json.loads(result.read_text())
    bad["output_manifest"][0]["sha256"] = "0" * 64
    result.write_text(json.dumps(bad), encoding="utf-8")
    try:
        controller._validate_downloaded_outputs(job, [(result, "file"), (image, "image")])
        raise AssertionError("corrupt checksum accepted")
    except RuntimeError as exc:
        assert "checksum invalide" in str(exc)
    controller.executor.shutdown(wait=False)

print("SHA-256 output manifest validation passed.")


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
    controller.db.create_job("export-sha", "image", "x", {})
    media_dir = settings.storage_root / "export-sha"
    media_dir.mkdir(parents=True)
    media = media_dir / "image.png"
    media.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    ))
    controller.db.add_artifact("export-sha", str(media), "image")
    archive = Path(controller.export_job_archive("export-sha"))
    import zipfile as _zipfile
    with _zipfile.ZipFile(archive) as zf:
        manifest = json.loads(zf.read("job.json"))
        exported = manifest["exported_artifacts"]
        assert len(exported) == 1
        assert exported[0]["name"] == "image.png"
        assert exported[0]["size"] == media.stat().st_size
        assert exported[0]["sha256"] == controller._hash_file(media)
        assert zf.read("media/image.png") == media.read_bytes()
    controller.executor.shutdown(wait=False)

print("Export SHA-256 manifest validation passed.")
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


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, keep_job_inputs=False,
        keep_source_inputs_for_retry=True, share_gradio=False,
    )
    seed_db = JobDB(settings.db_path)
    seed_db.create_job("resume-preparing", "image", "resume me", {
        "steps": 20, "cfg": 1.0, "seed": 1, "aspect": "1:1",
        "source_image": "", "target_video": "", "dataset_ref": "ci-user/stale-dataset",
    })
    seed_db.update_job("resume-preparing", status="preparing")

    original_execute = KaggleController._execute
    original_cleanup = KaggleController._cleanup_remote_refs
    old_user = os.environ.get("KAGGLE_USERNAME")
    old_token = os.environ.get("KAGGLE_API_TOKEN")
    executed = []
    cleaned = []
    KaggleController._execute = lambda self, job_id: executed.append(job_id)
    KaggleController._cleanup_remote_refs = lambda self, kernel_ref="", dataset_ref="", force=False: cleaned.append(
        (kernel_ref, dataset_ref, force)
    )
    try:
        os.environ["KAGGLE_USERNAME"] = "ci-user"
        os.environ["KAGGLE_API_TOKEN"] = "ci-token"
        controller = KaggleController(settings)
        controller.executor.shutdown(wait=True)
        recovered = controller.db.get_job("resume-preparing")
        assert recovered["status"] == "queued"
        assert recovered["meta"]["dataset_ref"] == ""
        assert executed == ["resume-preparing"]
        assert any(x[1] == "ci-user/stale-dataset" for x in cleaned)
    finally:
        KaggleController._execute = original_execute
        KaggleController._cleanup_remote_refs = original_cleanup
        if old_user is None:
            os.environ.pop("KAGGLE_USERNAME", None)
        else:
            os.environ["KAGGLE_USERNAME"] = old_user
        if old_token is None:
            os.environ.pop("KAGGLE_API_TOKEN", None)
        else:
            os.environ["KAGGLE_API_TOKEN"] = old_token

print("Persisted local preparation auto-resume passed.")


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
    controller.db.create_job("lost-done", "image", "x", {})
    controller.db.update_job("lost-done", status="done", kernel_ref="ci-user/still-complete")
    controller._kernel_status = lambda ref: ("complete", "complete")
    checked, recoverable = controller.reconcile_completed_jobs()
    job = controller.db.get_job("lost-done")
    assert checked == 1
    assert recoverable == 1
    assert job["status"] == "error"
    assert job["meta"]["recover_outputs_available"] is True
    assert job["meta"]["failed_phase"] == "local_artifacts_missing"
    controller.executor.shutdown(wait=False)

print("Lost completed output recovery detection passed.")


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
    controller.db.create_job("lost-unknown", "image", "x", {})
    controller.db.update_job("lost-unknown", status="done", kernel_ref="ci-user/gone")
    controller._kernel_status = lambda ref: (_ for _ in ()).throw(RuntimeError("404"))
    checked, recoverable = controller.reconcile_completed_jobs()
    job = controller.db.get_job("lost-unknown")
    assert checked == 1
    assert recoverable == 0
    assert job["status"] == "error"
    assert job["meta"].get("recover_outputs_available") is not True
    controller.executor.shutdown(wait=False)

print("Unconfirmed lost-output state validation passed.")


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
    image = tmp / "image.png"
    image.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    ))
    import hashlib as _hashlib
    digest = _hashlib.sha256(image.read_bytes()).hexdigest()
    result = tmp / "result.json"
    result.write_text(json.dumps({
        "status": "done", "files": ["image.png"],
        "output_manifest": [{"name": "image.png", "size": image.stat().st_size, "sha256": digest}],
    }), encoding="utf-8")
    job = {"task": "image", "meta": {}}
    parsed = controller._validate_downloaded_outputs(job, [(result, "file"), (image, "image")])
    assert parsed["output_manifest"][0]["sha256"] == digest
    bad = json.loads(result.read_text())
    bad["output_manifest"][0]["sha256"] = "0" * 64
    result.write_text(json.dumps(bad), encoding="utf-8")
    try:
        controller._validate_downloaded_outputs(job, [(result, "file"), (image, "image")])
        raise AssertionError("corrupt checksum accepted")
    except RuntimeError as exc:
        assert "checksum invalide" in str(exc)
    controller.executor.shutdown(wait=False)

print("SHA-256 output manifest validation passed.")


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
    controller.db.create_job("export-sha", "image", "x", {})
    media_dir = settings.storage_root / "export-sha"
    media_dir.mkdir(parents=True)
    media = media_dir / "image.png"
    media.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    ))
    controller.db.add_artifact("export-sha", str(media), "image")
    archive = Path(controller.export_job_archive("export-sha"))
    import zipfile as _zipfile
    with _zipfile.ZipFile(archive) as zf:
        manifest = json.loads(zf.read("job.json"))
        exported = manifest["exported_artifacts"]
        assert len(exported) == 1
        assert exported[0]["name"] == "image.png"
        assert exported[0]["size"] == media.stat().st_size
        assert exported[0]["sha256"] == controller._hash_file(media)
        assert zf.read("media/image.png") == media.read_bytes()
    controller.executor.shutdown(wait=False)

print("Export SHA-256 manifest validation passed.")


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
    image = tmp / "image.png"
    image.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    ))
    result = tmp / "result.json"
    result.write_text(json.dumps({
        "status": "done", "worker_version": "1.3", "files": ["image.png"],
    }), encoding="utf-8")
    try:
        controller._validate_downloaded_outputs(
            {"task": "image", "meta": {}}, [(result, "file"), (image, "image")]
        )
        raise AssertionError("v1.3 result without SHA-256 manifest accepted")
    except RuntimeError as exc:
        assert "Manifest SHA-256 absent" in str(exc)
    controller.executor.shutdown(wait=False)

print("Required SHA-256 manifest contract passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, keep_job_inputs=False,
        keep_source_inputs_for_retry=True, share_gradio=False,
    )
    seed_db = JobDB(settings.db_path)
    seed_db.create_job("auto-recover", "image", "x", {
        "recover_outputs_available": True, "auto_recovery_attempts": 1,
    })
    seed_db.update_job("auto-recover", status="error", kernel_ref="ci-user/complete")
    seed_db.create_job("auto-recover-capped", "image", "x", {
        "recover_outputs_available": True, "auto_recovery_attempts": 3,
    })
    seed_db.update_job("auto-recover-capped", status="error", kernel_ref="ci-user/complete-2")
    old_user = os.environ.get("KAGGLE_USERNAME")
    old_token = os.environ.get("KAGGLE_API_TOKEN")
    original_recover = KaggleController._recover_remote_job
    recovered_ids = []
    KaggleController._recover_remote_job = lambda self, job_id: recovered_ids.append(job_id)
    try:
        os.environ["KAGGLE_USERNAME"] = "ci-user"
        os.environ["KAGGLE_API_TOKEN"] = "ci-token"
        controller = KaggleController(settings)
        controller.executor.shutdown(wait=True)
        first = controller.db.get_job("auto-recover")
        capped = controller.db.get_job("auto-recover-capped")
        assert recovered_ids == ["auto-recover"]
        assert first["status"] == "recovering"
        assert first["meta"]["auto_recovery_attempts"] == 2
        assert first["meta"]["last_auto_recovery_at"] > 0
        assert capped["status"] == "error"
        assert capped["meta"]["auto_recovery_attempts"] == 3
    finally:
        KaggleController._recover_remote_job = original_recover
        if old_user is None:
            os.environ.pop("KAGGLE_USERNAME", None)
        else:
            os.environ["KAGGLE_USERNAME"] = old_user
        if old_token is None:
            os.environ.pop("KAGGLE_API_TOKEN", None)
        else:
            os.environ["KAGGLE_API_TOKEN"] = old_token

print("Automatic preserved-output recovery cap passed.")


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
    controller.db.create_job("missing-remote", "image", "x", {
        "recover_outputs_available": True, "auto_recovery_attempts": 1,
    })
    controller.db.update_job(
        "missing-remote", status="error", kernel_ref="ci-user/gone",
    )
    original_status = controller._kernel_status
    controller._kernel_status = lambda ref: (_ for _ in ()).throw(
        RuntimeError("404 Not Found: kernel does not exist")
    )
    controller._recover_remote_job("missing-remote")
    lost = controller.db.get_job("missing-remote")
    assert lost["status"] == "error"
    assert lost["meta"]["recover_outputs_available"] is False
    assert lost["meta"]["failed_phase"] == "remote_missing"
    assert "Relancer" in lost["error"]
    assert controller._remote_kernel_missing(RuntimeError("404 Not Found"))
    assert not controller._remote_kernel_missing(RuntimeError("429 Too Many Requests"))
    controller._kernel_status = original_status
    controller.executor.shutdown(wait=False)

print("Missing remote recovery source classification passed.")


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
    controller.db.create_job("delete-remote", "image", "x", {"dataset_ref": "ci-user/dataset"})
    controller.db.update_job("delete-remote", status="error", kernel_ref="ci-user/kernel")
    cleaned = []
    controller._cleanup_remote_refs = lambda kernel, dataset: cleaned.append((kernel, dataset))
    msg = controller.delete_local_job("delete-remote")
    assert cleaned == [("ci-user/kernel", "ci-user/dataset")]
    assert controller.db.get_job("delete-remote") is None
    assert "supprimé" in msg
    controller.executor.shutdown(wait=False)

print("Remote cleanup on local deletion passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, recovery_retention_days=7,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    old_user = os.environ.pop("KAGGLE_USERNAME", None)
    old_token = os.environ.pop("KAGGLE_API_TOKEN", None)
    old_key = os.environ.pop("KAGGLE_KEY", None)
    try:
        controller = KaggleController(settings)
        controller.db.create_job("login-recover", "image", "x", {
            "recover_outputs_available": True, "auto_recovery_attempts": 0,
        })
        controller.db.update_job(
            "login-recover", status="error", kernel_ref="ci-user/complete",
        )
        recovered_ids = []
        original_recover = controller._recover_remote_job
        controller._recover_remote_job = lambda job_id: recovered_ids.append(job_id)
        os.environ["KAGGLE_USERNAME"] = "ci-user"
        os.environ["KAGGLE_API_TOKEN"] = "ci-token"
        count = controller.resume_recoverable_outputs()
        controller.executor.shutdown(wait=True)
        row = controller.db.get_job("login-recover")
        assert count == 1
        assert recovered_ids == ["login-recover"]
        assert row["status"] == "recovering"
        assert row["meta"]["auto_recovery_attempts"] == 1
        assert row["meta"]["last_auto_recovery_at"] > 0
        controller._recover_remote_job = original_recover
    finally:
        if old_user is None:
            os.environ.pop("KAGGLE_USERNAME", None)
        else:
            os.environ["KAGGLE_USERNAME"] = old_user
        if old_token is None:
            os.environ.pop("KAGGLE_API_TOKEN", None)
        else:
            os.environ["KAGGLE_API_TOKEN"] = old_token
        if old_key is None:
            os.environ.pop("KAGGLE_KEY", None)
        else:
            os.environ["KAGGLE_KEY"] = old_key

print("Credential-time output recovery resume passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, recovery_retention_days=7,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    settings.env_file.write_text(
        "KAGGLE_ACCELERATOR=NvidiaTeslaP100\n"
        "QWEN_KAGGLE_SHARE=false\n"
        "KAGGLE_USERNAME=old-user\n"
        "KAGGLE_API_TOKEN=old-token\n",
        encoding="utf-8",
    )
    controller = KaggleController(settings)
    original_run = controller._run
    original_resume_wait = controller.resume_waiting_jobs
    original_resume_outputs = controller.resume_recoverable_outputs
    controller._run = lambda *args, **kwargs: "ok"
    controller.resume_waiting_jobs = lambda: 0
    controller.resume_recoverable_outputs = lambda: 0
    old_user = os.environ.get("KAGGLE_USERNAME")
    old_token = os.environ.get("KAGGLE_API_TOKEN")
    old_key = os.environ.get("KAGGLE_KEY")
    try:
        controller.save_credentials("new-user", "new-token", "", persist=True)
        saved = settings.env_file.read_text(encoding="utf-8")
        assert "KAGGLE_ACCELERATOR=NvidiaTeslaP100" in saved
        assert "QWEN_KAGGLE_SHARE=false" in saved
        assert "KAGGLE_USERNAME=new-user" in saved
        assert "KAGGLE_API_TOKEN=new-token" in saved
        assert "old-user" not in saved
        assert "old-token" not in saved
        assert "KAGGLE_KEY=" not in saved
    finally:
        controller._run = original_run
        controller.resume_waiting_jobs = original_resume_wait
        controller.resume_recoverable_outputs = original_resume_outputs
        controller.executor.shutdown(wait=False)
        if old_user is None:
            os.environ.pop("KAGGLE_USERNAME", None)
        else:
            os.environ["KAGGLE_USERNAME"] = old_user
        if old_token is None:
            os.environ.pop("KAGGLE_API_TOKEN", None)
        else:
            os.environ["KAGGLE_API_TOKEN"] = old_token
        if old_key is None:
            os.environ.pop("KAGGLE_KEY", None)
        else:
            os.environ["KAGGLE_KEY"] = old_key

print("Credential persistence preserves settings passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, recovery_retention_days=7,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    original_run = controller._run
    original_resume_wait = controller.resume_waiting_jobs
    original_resume_outputs = controller.resume_recoverable_outputs
    controller._run = lambda *args, **kwargs: "ok"
    controller.resume_waiting_jobs = lambda: 0
    controller.resume_recoverable_outputs = lambda: 0
    old_user = os.environ.get("KAGGLE_USERNAME")
    old_token = os.environ.get("KAGGLE_API_TOKEN")
    old_key = os.environ.get("KAGGLE_KEY")
    try:
        controller.save_credentials("atomic-user", "atomic-token", "", persist=True)
        assert settings.env_file.exists()
        saved = settings.env_file.read_text(encoding="utf-8")
        assert "KAGGLE_USERNAME=atomic-user" in saved
        assert "KAGGLE_API_TOKEN=atomic-token" in saved
        assert not list(settings.env_file.parent.glob(f".{settings.env_file.name}.*.part"))
    finally:
        controller._run = original_run
        controller.resume_waiting_jobs = original_resume_wait
        controller.resume_recoverable_outputs = original_resume_outputs
        controller.executor.shutdown(wait=False)
        if old_user is None:
            os.environ.pop("KAGGLE_USERNAME", None)
        else:
            os.environ["KAGGLE_USERNAME"] = old_user
        if old_token is None:
            os.environ.pop("KAGGLE_API_TOKEN", None)
        else:
            os.environ["KAGGLE_API_TOKEN"] = old_token
        if old_key is None:
            os.environ.pop("KAGGLE_KEY", None)
        else:
            os.environ["KAGGLE_KEY"] = old_key

print("Atomic credential persistence passed.")


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
    original_run = controller._run
    controller._run = lambda *args, **kwargs: "ok"
    bad_cases = [
        ("user\\nEVIL=1", "token", ""),
        ("user", "token\\nEVIL=1", ""),
        ("user", "", "key\\rEVIL=1"),
        ("bad user", "token", ""),
    ]
    for username, token, key in bad_cases:
        try:
            controller.save_credentials(username, token, key, persist=True)
            raise AssertionError("unsafe credential value accepted")
        except ValueError:
            pass
    assert not settings.env_file.exists()
    controller._run = original_run
    controller.executor.shutdown(wait=False)

print("Credential env injection protection passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    cleanup_commands = []
    controller._run = lambda args, **kwargs: cleanup_commands.append(args) or ""
    controller._cleanup_remote_refs("ci-user/keep-kernel", "ci-user/temp-dataset")
    assert ["kernels", "delete", "ci-user/keep-kernel", "-y"] not in cleanup_commands
    assert ["datasets", "delete", "ci-user/temp-dataset", "-y"] in cleanup_commands
    controller.executor.shutdown(wait=False)

print("Independent Kaggle dataset cleanup policy passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=False,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    cleanup_commands = []
    controller._run = lambda args, **kwargs: cleanup_commands.append(args) or ""
    controller._cleanup_remote_refs("ci-user/kernel", "ci-user/dataset", force=True)
    assert ["kernels", "delete", "ci-user/kernel", "-y"] in cleanup_commands
    assert ["datasets", "delete", "ci-user/dataset", "-y"] in cleanup_commands
    controller.executor.shutdown(wait=False)

print("Forced Kaggle remote cleanup override passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    settings.env_file.write_text(
        "KAGGLE_USERNAME=working-user\n"
        "KAGGLE_API_TOKEN=working-token\n"
        "QWEN_KAGGLE_SHARE=false\n",
        encoding="utf-8",
    )
    old_values = {key: os.environ.get(key) for key in (
        "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY"
    )}
    os.environ["KAGGLE_USERNAME"] = "working-user"
    os.environ["KAGGLE_API_TOKEN"] = "working-token"
    os.environ.pop("KAGGLE_KEY", None)
    controller = KaggleController(settings)
    controller._run = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("401 Unauthorized")
    )
    before_file = settings.env_file.read_text(encoding="utf-8")
    try:
        controller.save_credentials("bad-user", "bad-token", "", persist=True)
        raise AssertionError("invalid credentials unexpectedly persisted")
    except RuntimeError as exc:
        assert "401" in str(exc)
    assert os.environ.get("KAGGLE_USERNAME") == "working-user"
    assert os.environ.get("KAGGLE_API_TOKEN") == "working-token"
    assert "KAGGLE_KEY" not in os.environ
    assert settings.env_file.read_text(encoding="utf-8") == before_file
    controller.executor.shutdown(wait=False)
    for key, value in old_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

print("Invalid Kaggle credential rollback passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=True, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.db.create_job("ambiguous-remote", "image", "x", {})
    controller._prepare_kernel = lambda job, folder, dataset_ref: "ci-user/ambiguous-kernel"
    controller._run = lambda *args, **kwargs: ""
    controller._kernel_status = lambda ref: (_ for _ in ()).throw(
        RuntimeError("temporary status endpoint failure")
    )
    cleanup_calls = []
    controller._cleanup_remote_refs = lambda kernel_ref="", dataset_ref="", force=False: cleanup_calls.append(
        (kernel_ref, dataset_ref, force)
    )
    controller._execute("ambiguous-remote")
    row = controller.db.get_job("ambiguous-remote")
    assert row["status"] == "error"
    assert row["meta"]["recover_outputs_available"] is True
    assert row["meta"]["remote_failure_confirmed"] is False
    assert row["kernel_ref"] == "ci-user/ambiguous-kernel"
    assert not any(call[0] == "ci-user/ambiguous-kernel" for call in cleanup_calls)
    controller.executor.shutdown(wait=False)

print("Ambiguous remote failure preserves kernel for recovery.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=True, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.db.create_job("confirmed-remote-failure", "image", "x", {})
    controller._prepare_kernel = lambda job, folder, dataset_ref: "ci-user/failed-kernel"
    command_log = []
    controller._run = lambda args, **kwargs: command_log.append(args) or (
        "worker traceback" if args[:2] == ["kernels", "logs"] else ""
    )
    controller._kernel_status = lambda ref: ("error", "status: error")
    cleanup_calls = []
    controller._cleanup_remote_refs = lambda kernel_ref="", dataset_ref="", force=False: cleanup_calls.append(
        (kernel_ref, dataset_ref, force)
    )
    controller._execute("confirmed-remote-failure")
    row = controller.db.get_job("confirmed-remote-failure")
    assert row["status"] == "error"
    assert row["meta"]["recover_outputs_available"] is False
    assert row["meta"]["remote_failure_confirmed"] is True
    assert any(call[0] == "ci-user/failed-kernel" for call in cleanup_calls)
    controller.executor.shutdown(wait=False)

print("Confirmed remote failure does not create false recovery source.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    controller._run = lambda *args, **kwargs: "mystery-state-from-api"
    try:
        controller._kernel_status("ci-user/kernel")
        raise AssertionError("unknown Kaggle status was treated as running")
    except RuntimeError as exc:
        assert "Statut Kaggle non reconnu" in str(exc)
    controller.executor.shutdown(wait=False)

print("Unknown Kaggle status is fail-safe instead of assumed running.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.db.create_job("download-retry", "image", "x", {})
    job = controller.db.get_job("download-retry")
    calls = {"count": 0}
    original_sleep = time.sleep

    def fake_output(args, **kwargs):
        if args[:2] != ["kernels", "output"]:
            return ""
        calls["count"] += 1
        out_dir = Path(args[args.index("-p") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        if calls["count"] == 1:
            (out_dir / "stale.bin").write_bytes(b"partial")
            (out_dir / "result.json").write_text(
                json.dumps({"status": "done", "files": ["image.png"]}),
                encoding="utf-8",
            )
            return ""
        (out_dir / "image.png").write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        ))
        (out_dir / "result.json").write_text(
            json.dumps({"status": "done", "files": ["image.png"]}),
            encoding="utf-8",
        )
        return ""

    controller._run = fake_output
    try:
        time.sleep = lambda *_args, **_kwargs: None
        download = tmp / "download"
        staged, manifest = controller._download_validated_outputs(
            job, "ci-user/kernel", download, attempts=2
        )
    finally:
        time.sleep = original_sleep
    assert calls["count"] == 2
    assert manifest["status"] == "done"
    assert (download / "image.png").is_file()
    assert not (download / "stale.bin").exists()
    assert any(path.name == "image.png" and kind == "image" for path, kind in staged)
    controller.executor.shutdown(wait=False)

print("Validated Kaggle output retry from clean directory passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    controller.db.create_job("download-exhausted", "image", "x", {})
    job = controller.db.get_job("download-exhausted")
    calls = {"count": 0}
    original_sleep = time.sleep

    def always_partial(args, **kwargs):
        if args[:2] == ["kernels", "output"]:
            calls["count"] += 1
            out_dir = Path(args[args.index("-p") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "result.json").write_text(
                json.dumps({"status": "done", "files": ["missing.png"]}),
                encoding="utf-8",
            )
        return ""

    controller._run = always_partial
    try:
        time.sleep = lambda *_args, **_kwargs: None
        try:
            controller._download_validated_outputs(
                job, "ci-user/kernel", tmp / "download", attempts=3
            )
            raise AssertionError("incomplete Kaggle output accepted")
        except RuntimeError as exc:
            assert "après 3 tentative" in str(exc)
    finally:
        time.sleep = original_sleep
    assert calls["count"] == 3
    controller.executor.shutdown(wait=False)

print("Exhausted Kaggle output retries fail closed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    seed = JobDB(settings.db_path)
    seed.create_job("remote-waits-auth", "image", "x", {})
    seed.update_job(
        "remote-waits-auth", status="running", kernel_ref="ci-user/existing-kernel"
    )
    old_values = {key: os.environ.pop(key, None) for key in (
        "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY"
    )}
    try:
        controller = KaggleController(settings)
        controller.executor.shutdown(wait=True)
        row = controller.db.get_job("remote-waits-auth")
        assert row["status"] == "waiting_auth"
        assert row["kernel_ref"] == "ci-user/existing-kernel"
        assert "kernel distant existant" in row["error"]
    finally:
        for key, value in old_values.items():
            if value is not None:
                os.environ[key] = value

print("Remote kernel waits safely for authentication after restart.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    old_values = {key: os.environ.pop(key, None) for key in (
        "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY"
    )}
    try:
        controller = KaggleController(settings)
        controller.db.create_job("resume-existing-kernel", "image", "x", {})
        controller.db.update_job(
            "resume-existing-kernel",
            status="waiting_auth",
            kernel_ref="ci-user/existing-kernel",
        )
        recovered = []
        executed = []
        controller._recover_remote_job = lambda job_id: recovered.append(job_id)
        controller._execute = lambda job_id: executed.append(job_id)
        os.environ["KAGGLE_USERNAME"] = "ci-user"
        os.environ["KAGGLE_API_TOKEN"] = "ci-token"
        count = controller.resume_waiting_jobs()
        controller.executor.shutdown(wait=True)
        row = controller.db.get_job("resume-existing-kernel")
        assert count == 1
        assert recovered == ["resume-existing-kernel"]
        assert executed == []
        assert row["status"] == "recovering"
        assert row["kernel_ref"] == "ci-user/existing-kernel"
    finally:
        for key in ("KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY"):
            os.environ.pop(key, None)
        for key, value in old_values.items():
            if value is not None:
                os.environ[key] = value

print("Authentication resumes existing remote kernel without recompute.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)
    active_cases = [
        ("retry-preparing", "preparing", ""),
        ("retry-uploading", "uploading_inputs", ""),
        ("retry-waiting-local", "waiting_auth", ""),
        ("retry-waiting-remote", "waiting_auth", "ci-user/existing"),
        ("retry-cancel-requested", "cancel_requested", "ci-user/existing"),
    ]
    for job_id, status, kernel_ref in active_cases:
        controller.db.create_job(job_id, "image", "x", {})
        controller.db.update_job(job_id, status=status, kernel_ref=kernel_ref)
        try:
            controller.retry(job_id)
            raise AssertionError(f"active job retry unexpectedly accepted: {status}")
        except RuntimeError as exc:
            text = str(exc).lower()
            assert "actif" in text or "authentification" in text
    controller.executor.shutdown(wait=False)

print("Active Kaggle job duplicate retry guard passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)

    cases = [
        ("status: running", "running"),
        ("Kernel Status = completed", "complete"),
        ("queued", "queued"),
        ("FAILED", "error"),
    ]
    for raw, expected in cases:
        controller._run = lambda *args, _raw=raw, **kwargs: _raw
        state, returned = controller._kernel_status("ci-user/kernel")
        assert state == expected
        assert returned == raw

    ambiguous = [
        "running\nlast completed version: 4",
        "status unknown; no error detected",
        "completed build metadata\nrunning now",
    ]
    for raw in ambiguous:
        controller._run = lambda *args, _raw=raw, **kwargs: _raw
        try:
            controller._kernel_status("ci-user/kernel")
            raise AssertionError(f"ambiguous status unexpectedly accepted: {raw}")
        except RuntimeError as exc:
            message = str(exc)
            assert (
                "Statut Kaggle non reconnu" in message
                or "Statut Kaggle ambigu" in message
            )

    controller.executor.shutdown(wait=False)

print("Kaggle status parser ambiguity protection passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    old_values = {key: os.environ.get(key) for key in (
        "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY"
    )}
    os.environ["KAGGLE_USERNAME"] = "loaded-user"
    os.environ["KAGGLE_API_TOKEN"] = "loaded-token"
    os.environ.pop("KAGGLE_KEY", None)
    controller = KaggleController(settings)
    original_run = controller._run
    original_resume_wait = controller.resume_waiting_jobs
    original_resume_outputs = controller.resume_recoverable_outputs
    controller._run = lambda *args, **kwargs: "ok"
    controller.resume_waiting_jobs = lambda: 0
    controller.resume_recoverable_outputs = lambda: 0
    try:
        msg = controller.save_credentials("loaded-user", "", "", persist=True)
        saved = settings.env_file.read_text(encoding="utf-8")
        assert "KAGGLE_USERNAME=loaded-user" in saved
        assert "KAGGLE_API_TOKEN=loaded-token" in saved
        assert "KAGGLE_KEY=" not in saved
        assert "validés" in msg
    finally:
        controller._run = original_run
        controller.resume_waiting_jobs = original_resume_wait
        controller.resume_recoverable_outputs = original_resume_outputs
        controller.executor.shutdown(wait=False)
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

print("Loaded Kaggle secret reuse without UI exposure passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    old_values = {key: os.environ.get(key) for key in (
        "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY"
    )}
    os.environ["KAGGLE_USERNAME"] = "loaded-user"
    os.environ["KAGGLE_API_TOKEN"] = "loaded-token"
    os.environ.pop("KAGGLE_KEY", None)
    controller = KaggleController(settings)
    try:
        controller.save_credentials("different-user", "", "", persist=False)
        raise AssertionError("secret from another username was reused")
    except ValueError:
        pass
    finally:
        controller.executor.shutdown(wait=False)
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

print("Loaded Kaggle secret is not reused for a different username.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    controller = KaggleController(settings)

    cases = [
        ("status: ready", "ready"),
        ("Dataset Status = completed", "ready"),
        ("pending", "pending"),
        ("FAILED", "error"),
    ]
    for raw, expected in cases:
        controller._run = lambda *args, _raw=raw, **kwargs: _raw
        state, returned = controller._dataset_status("ci-user/dataset")
        assert state == expected
        assert returned == raw

    ambiguous = [
        "pending\nlast completed version: 2",
        "status unknown; no error detected",
        "ready cache metadata\ncreating now",
    ]
    for raw in ambiguous:
        controller._run = lambda *args, _raw=raw, **kwargs: _raw
        state, returned = controller._dataset_status("ci-user/dataset")
        assert state == "unknown"
        assert returned == raw

    controller.executor.shutdown(wait=False)

print("Kaggle dataset status parser ambiguity protection passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    old_values = {key: os.environ.get(key) for key in (
        "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY"
    )}
    os.environ["KAGGLE_USERNAME"] = "cache-user"
    os.environ["KAGGLE_API_TOKEN"] = "cache-token"
    os.environ.pop("KAGGLE_KEY", None)
    controller = KaggleController(settings)
    calls = {"count": 0}
    controller._run = lambda *args, **kwargs: calls.__setitem__("count", calls["count"] + 1) or "ok"
    try:
        controller.validate_current_credentials(max_age_seconds=300)
        controller.validate_current_credentials(max_age_seconds=300)
        assert calls["count"] == 1

        os.environ["KAGGLE_API_TOKEN"] = "changed-token"
        controller.validate_current_credentials(max_age_seconds=300)
        assert calls["count"] == 2
    finally:
        controller.executor.shutdown(wait=False)
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

print("Kaggle credential validation cache and fingerprint invalidation passed.")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    settings = Settings(
        root=repo_root, storage_root=tmp / "media", db_path=tmp / "jobs.sqlite3",
        env_file=tmp / ".env.local", worker_path=repo_root / "kaggle_worker" / "worker.py",
        poll_seconds=1, cli_retries=1, kernel_timeout=60, accelerator="NvidiaTeslaT4",
        delete_remote_kernel=False, delete_remote_dataset=True,
        keep_job_inputs=False, keep_source_inputs_for_retry=True, share_gradio=False,
    )
    old_values = {key: os.environ.get(key) for key in (
        "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY"
    )}
    os.environ["KAGGLE_USERNAME"] = "bad-user"
    os.environ["KAGGLE_API_TOKEN"] = "expired-token"
    os.environ.pop("KAGGLE_KEY", None)
    controller = KaggleController(settings)
    controller._run = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("401 Unauthorized")
    )
    before = len(controller.jobs(100))
    try:
        controller.submit("image", "test prompt")
        raise AssertionError("job created with invalid Kaggle credentials")
    except RuntimeError as exc:
        assert "401" in str(exc)
    assert len(controller.jobs(100)) == before
    controller.executor.shutdown(wait=False)
    for key, value in old_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

print("Invalid Kaggle credentials are rejected before job persistence.")

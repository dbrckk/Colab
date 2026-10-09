"""Controller-level persistence smoke test, using in-memory private object storage."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from kaggle_app.config import Settings
from kaggle_app.kaggle_runner import KaggleController
from kaggle_app.remote_storage import RemoteStorage


class FakeStore(RemoteStorage):
    def __init__(self):
        super().__init__("https://example.supabase.co", "test-server-key")
        self.objects: dict[str, bytes] = {}

    def ensure_private_bucket(self) -> None:
        pass

    def object_get(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def object_put(self, key: str, content: bytes, mime: str) -> None:
        self.objects[key] = bytes(content)

    def object_remove(self, keys: list[str]) -> None:
        if getattr(self, "fail_remove_once", False):
            self.fail_remove_once = False
            raise RuntimeError("Simulated Supabase delete outage")
        for key in keys:
            self.objects.pop(key, None)


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    storage = root / "media"
    db = root / "db" / "jobs.sqlite3"
    settings = Settings(
        root=root,
        storage_root=storage,
        db_path=db,
        env_file=root / ".env.local",
        worker_path=root / "worker.py",
        share_gradio=False,
    )
    fake = FakeStore()
    with patch("kaggle_app.kaggle_runner.RemoteStorage.from_env", return_value=fake):
        app = KaggleController(settings)

    source = root / "source.png"
    source.write_bytes(b"fake-image-input-contents")
    saved_source = app._persist_input("job1", str(source), "source_image")
    assert Path(saved_source).exists()

    app.db.create_job("job1", "image_edit", "enhance", {
        "source_image": saved_source, "target_video": "",
    })
    output_dir = root / "download"
    output_dir.mkdir()
    (output_dir / "image.png").write_bytes(b"fake-image-output-contents")
    app._commit_downloaded_outputs(app.db.get_job("job1"), output_dir, {})
    media = Path(app.db.artifacts("job1")[0]["path"])
    assert media.is_file()
    assert app.db.get_job("job1")["status"] == "done"

    # Simulate a Render cold start with a wiped filesystem.
    media.unlink()
    Path(saved_source).unlink()
    db.unlink()

    with patch("kaggle_app.kaggle_runner.RemoteStorage.from_env", return_value=fake):
        restored = KaggleController(settings)
    assert restored.db.get_job("job1")["status"] == "done"
    assert len(restored.db.artifacts("job1")) == 1
    assert not media.exists()  # lazy restoration avoids blocking startup
    assert restored.artifacts("job1")[0]["path"] == str(media)
    assert media.read_bytes() == b"fake-image-output-contents"
    assert restored.recent_artifacts("image", 10)[0]["path"] == str(media)

    # Source inputs return when the user loads/retries a saved job.
    restored._restore_required_inputs(restored.db.get_job("job1"))
    assert Path(saved_source).read_bytes() == b"fake-image-input-contents"

    # Delete after another simulated loss of the local disk. The remote
    # source still needs to be removed even if no local input cache exists.
    media.unlink()
    Path(saved_source).unlink()
    fake.fail_remove_once = True
    msg = restored.delete_local_job("job1")
    assert "supprimé" in msg and "en attente" in msg
    assert restored.db.get_job("job1") is None
    assert not media.exists()
    assert restored.db.pending_remote_deletes(), "Failed remote deletion lost its tombstone"

    # Next cold start must finish the private remote deletion and preserve
    # the deleted state, without resurrecting stale gallery records.
    db.unlink()
    with patch("kaggle_app.kaggle_runner.RemoteStorage.from_env", return_value=fake):
        after_delete = KaggleController(settings)
    assert after_delete.db.get_job("job1") is None
    assert after_delete.db.pending_remote_deletes() == []
    assert fake._manifest("job1/image.png") is None
    assert fake._manifest("_input_store/" + Path(saved_source).name) is None

    app.executor.shutdown(wait=True)
    restored.executor.shutdown(wait=True)
    after_delete.executor.shutdown(wait=True)

print("Controller remote-persistence integration smoke passed.")

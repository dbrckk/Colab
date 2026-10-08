"""Offline tests for private remote persistence; no Supabase credentials required."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from kaggle_app.db import JobDB
from kaggle_app.remote_storage import (
    CHUNK_SIZE,
    DB_OBJECT,
    RemoteStorage,
    RemoteStorageError,
)


class MemoryStore(RemoteStorage):
    """In-memory Supabase Storage substitute exercising the real storage logic."""

    def __init__(self):
        super().__init__("https://test.supabase.co", "local-test-secret")
        self.objects: dict[str, bytes] = {}
        self.failed_key = ""
        self.bucket_verified = False

    def ensure_private_bucket(self) -> None:
        self.bucket_verified = True

    def object_get(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def object_put(self, key: str, content: bytes, mime: str) -> None:
        if self.failed_key and self.failed_key in key:
            raise RemoteStorageError("Mock storage unavailable")
        self.objects[key] = bytes(content)

    def object_remove(self, keys: list[str]) -> None:
        for key in keys:
            self.objects.pop(key, None)


def expect_error(fn, exception=RemoteStorageError) -> None:
    try:
        fn()
    except exception:
        pass
    else:
        raise AssertionError(f"Expected {exception.__name__}")


with patch.dict(os.environ, {
    "QWEN_SUPABASE_URL": "", "QWEN_SUPABASE_SERVICE_ROLE_KEY": "",
}):
    assert RemoteStorage.from_env() is None

with patch.dict(os.environ, {
    "QWEN_SUPABASE_URL": "https://example.supabase.co",
    "QWEN_SUPABASE_SERVICE_ROLE_KEY": "",
}):
    expect_error(RemoteStorage.from_env)
expect_error(lambda: RemoteStorage("http://insecure.example", "key"))
expect_error(lambda: RemoteStorage("https://a.example", "key", bucket="Public!"))

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    store = MemoryStore()
    media = root / "media"
    original = media / "job-1" / "long-video.mp4"
    original.parent.mkdir(parents=True)
    content = b"A" * CHUNK_SIZE + b"B" * 200_003
    original.write_bytes(content)

    store.save_file(original, media)
    path_id = store._object_prefix("job-1/long-video.mp4")
    manifest = json.loads(store.objects[f"{path_id}/manifest.json"])
    assert manifest["size"] == len(content)
    assert len(manifest["parts"]) == 2
    assert max(part["size"] for part in manifest["parts"]) <= CHUNK_SIZE

    object_count = len(store.objects)
    store.save_file(original, media)
    assert len(store.objects) == object_count  # idempotent
    original.unlink()
    assert store.restore_file(original, media)
    assert original.read_bytes() == content
    assert not list(original.parent.glob(".restore-*"))

    # Corrupted/missing chunks can never replace a local file with bad data.
    original.unlink()
    key = manifest["parts"][-1]["key"]
    store.objects[key] = b"corrupted"
    expect_error(lambda: store.restore_file(original, media))
    assert not original.exists()
    assert not list(original.parent.glob(".restore-*"))
    store.objects[key] = b"B" * 200_003
    assert store.restore_file(original, media)
    assert original.read_bytes() == content

    outside = root / "outside.txt"
    outside.write_text("private")
    expect_error(lambda: store.save_file(outside, media))

    second = media / "job-2" / "partial.png"
    second.parent.mkdir(parents=True)
    second.write_bytes(b"image")
    store.failed_key = "/versions/"
    expect_error(lambda: store.save_file(second, media))
    assert store._manifest("job-2/partial.png") is None
    store.failed_key = ""
    store.save_file(second, media)
    store.remove_file(second, media)
    assert store._manifest("job-2/partial.png") is None

    assert not store.restore_file(media / "missing.mp4", media)

    db_store = MemoryStore()
    first = JobDB(root / "first.sqlite3", remote_store=db_store)
    first.create_job("job-1", "image", "prompt", {"seed": 11})
    first.update_job("job-1", status="done", kernel_ref="user/kernel")
    first.add_artifact("job-1", str(original), "video")
    assert DB_OBJECT in db_store.objects

    # A new Render instance starts on an empty local disk and restores state.
    second_db = JobDB(root / "second.sqlite3", remote_store=db_store)
    assert second_db.get_job("job-1")["status"] == "done"
    assert second_db.get_job("job-1")["kernel_ref"] == "user/kernel"
    assert len(second_db.artifacts("job-1")) == 1
    second_db.delete_artifacts_by_ids([second_db.artifacts("job-1")[0]["id"]])
    third_db = JobDB(root / "third.sqlite3", remote_store=db_store)
    assert third_db.artifacts("job-1") == []
    third_db.delete_job("job-1")
    fourth_db = JobDB(root / "fourth.sqlite3", remote_store=db_store)
    assert fourth_db.get_job("job-1") is None

    # Unreadable backups must fail closed, not wipe a valid live database.
    db_store.objects[DB_OBJECT] = b"broken"
    expect_error(lambda: JobDB(root / "first.sqlite3", remote_store=db_store))
    assert (root / "first.sqlite3").exists()

    # Upload outages are reported, not silently reported as persisted.
    other = MemoryStore()
    volatile = JobDB(root / "volatile.sqlite3", remote_store=other)
    other.failed_key = DB_OBJECT
    expect_error(lambda: volatile.create_job("x", "image", "hi", {}))

print("Remote persistent storage unit tests passed.")

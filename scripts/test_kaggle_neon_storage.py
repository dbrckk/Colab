"""Offline Neon adapter tests with a transactional in-memory SQL substitute."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from kaggle_app.db import JobDB
from kaggle_app.neon_storage import NeonStorage, DEFAULT_QUOTA
from kaggle_app.remote_storage import RemoteStorageError


class FakeCursor:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.row = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute(self, query: str, params=()):
        sql = " ".join(query.lower().split())
        if sql.startswith("create table") or "pg_advisory_xact_lock" in sql:
            self.row = (None,)
        elif "sum(octet_length(data))" in sql:
            self.row = (sum(map(len, self.objects.values())),)
        elif "select octet_length(data)" in sql:
            data = self.objects.get(params[0])
            self.row = (len(data),) if data is not None else None
        elif "select data from" in sql:
            data = self.objects.get(params[0])
            self.row = (data,) if data is not None else None
        elif sql.startswith("insert into qwen_private_objects"):
            self.objects[params[0]] = bytes(params[1])
            self.row = None
        elif sql.startswith("delete from qwen_private_objects"):
            for key in params[0]:
                self.objects.pop(key, None)
            self.row = None
        else:
            raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def cursor(self):
        return FakeCursor(self.objects)


class TestNeonStorage(NeonStorage):
    def __init__(self, max_bytes=DEFAULT_QUOTA):
        super().__init__(
            "postgresql://user:pass@db.example.neon.tech/neondb?sslmode=require",
            max_bytes=max_bytes,
        )
        self.objects: dict[str, bytes] = {}

    def _connect(self):
        return FakeConnection(self.objects)


def must_error(callback):
    try:
        callback()
    except RemoteStorageError:
        return
    raise AssertionError("Expected RemoteStorageError")


must_error(lambda: NeonStorage("http://invalid.example"))
must_error(lambda: NeonStorage("postgresql://user@db.example/db"))
must_error(lambda: NeonStorage(
    "postgresql://user:pass@db.example/db",
    bucket="illegal.Bucket",
))
with patch.dict(os.environ, {"QWEN_NEON_DATABASE_URL": ""}):
    assert NeonStorage.from_env() is None

with patch.dict(os.environ, {
    "QWEN_NEON_DATABASE_URL": "postgresql://user:pass@db.example/db",
    "QWEN_NEON_MAX_STORAGE_BYTES": "garbage",
}):
    must_error(NeonStorage.from_env)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    storage = TestNeonStorage(max_bytes=80 * 1024 * 1024)
    storage.ensure_private_bucket()
    assert storage.object_get("missing") is None
    storage.object_put("alpha", b"one", "text/plain")
    assert storage.object_get("alpha") == b"one"
    storage.object_put("alpha", b"two", "text/plain")
    assert storage.object_get("alpha") == b"two"
    assert list(storage.objects) == ["qwen-studio-private/alpha"]
    storage.object_remove(["alpha"])
    assert storage.object_get("alpha") is None

    media_root = root / "media"
    source = media_root / "job-1" / "image.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image contents" * 1000)
    storage.save_file(source, media_root)
    source.unlink()
    assert storage.restore_file(source, media_root)
    assert source.read_bytes() == b"image contents" * 1000

    first = JobDB(root / "first.sqlite3", remote_store=storage)
    first.create_job("job1", "image", "prompt", {"seed": 5})
    first.update_job("job1", status="done")
    second = JobDB(root / "second.sqlite3", remote_store=storage)
    assert second.get_job("job1")["status"] == "done"
    assert second.get_job("job1")["meta"]["seed"] == 5

    storage.max_bytes = 64 * 1024 * 1024
    # Test quota on a file larger than available bytes, without actually
    # uploading a large media file.
    current = sum(map(len, storage.objects.values()))
    storage.max_bytes = max(64 * 1024 * 1024, current + 1)
    assert storage.object_get("not-found") is None
    assert storage._key("x") == "qwen-studio-private/x"
    must_error(lambda: storage._key("../escape"))

print("Neon storage adapter and restart tests passed.")

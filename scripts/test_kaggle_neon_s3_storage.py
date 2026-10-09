"""Offline Neon S3 tests: access control, verified media and cold start."""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from kaggle_app.db import JobDB
from kaggle_app.neon_s3_storage import NeonObjectStorage
from kaggle_app.remote_storage import RemoteStorageError, CHUNK_SIZE


class MissingS3Object(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.grants = []
        self.fail_write = False

    def head_bucket(self, **kwargs):
        assert kwargs["Bucket"] == "qwen-studio-private"

    def get_bucket_acl(self, **kwargs):
        return {"Grants": self.grants}

    def put_object(self, **kwargs):
        if self.fail_write:
            raise RuntimeError("S3 outage: must be masked")
        self.objects[kwargs["Key"]] = bytes(kwargs["Body"])

    def get_object(self, **kwargs):
        data = self.objects.get(kwargs["Key"])
        if data is None:
            raise MissingS3Object()
        return {"Body": io.BytesIO(data)}

    def delete_objects(self, **kwargs):
        for item in kwargs["Delete"]["Objects"]:
            self.objects.pop(item["Key"], None)
        return {"Errors": []}


def fails(action):
    try:
        action()
    except RemoteStorageError:
        return
    raise AssertionError("Expected RemoteStorageError")


TEST_ENDPOINT = "https://br-test.storage.c-5.eu-central-1.aws.neon.tech"


def new_store(s3):
    return NeonObjectStorage(
        TEST_ENDPOINT,
        "nak_test-access",
        "nsk_test-secret",
        "eu-central-1",
        client=s3,
    )


with patch.dict(os.environ, {
    "QWEN_NEON_S3_ENDPOINT": "",
    "QWEN_NEON_S3_ACCESS_KEY_ID": "",
    "QWEN_NEON_S3_SECRET_ACCESS_KEY": "",
    "QWEN_NEON_S3_REGION": "",
}):
    assert NeonObjectStorage.from_env() is None

with patch.dict(os.environ, {
    "QWEN_NEON_S3_ENDPOINT": TEST_ENDPOINT,
    "QWEN_NEON_S3_ACCESS_KEY_ID": "",
    "QWEN_NEON_S3_SECRET_ACCESS_KEY": "",
    "QWEN_NEON_S3_REGION": "",
}):
    fails(NeonObjectStorage.from_env)

fails(lambda: NeonObjectStorage(
    "http://insecure.example", "nak_a", "nsk_b", "eu-central-1",
    client=FakeS3(),
))
fails(lambda: NeonObjectStorage(
    "https://attacker.example", "nak_a", "nsk_b", "eu-central-1",
    client=FakeS3(),
))
fails(lambda: NeonObjectStorage(
    TEST_ENDPOINT, "other", "wrong", "eu-central-1",
    client=FakeS3(),
))

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    media_root = root / "media"
    s3 = FakeS3()
    remote = new_store(s3)
    remote.ensure_private_bucket()

    s3.grants = [{
        "Grantee": {
            "URI": "http://acs.amazonaws.com/groups/global/AllUsers"
        }
    }]
    fails(remote.ensure_private_bucket)
    s3.grants = []
    remote.ensure_private_bucket()

    path = media_root / "job-one" / "video.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = b"A" * CHUNK_SIZE + b"B" * 23
    path.write_bytes(content)
    remote.save_file(path, media_root)
    count = len(s3.objects)
    remote.save_file(path, media_root)
    assert len(s3.objects) == count
    path.unlink()
    assert remote.restore_file(path, media_root)
    assert path.read_bytes() == content

    path.unlink()
    key = next(key for key in s3.objects if key.endswith("/000001.bin"))
    s3.objects[key] = b"corrupt"
    fails(lambda: remote.restore_file(path, media_root))
    assert not path.exists()
    s3.objects[key] = b"B" * 23

    # SQLite snapshots are saved on each mutation, then restored on a clean
    # machine exactly as the real Render controller does.
    a = JobDB(root / "db-a.sqlite3", remote_store=remote)
    a.create_job("job-1", "image", "mountain", {"steps": 20})
    a.add_artifact("job-1", str(path), "video")
    a.update_job("job-1", status="done")

    b = JobDB(root / "db-b.sqlite3", remote_store=remote)
    assert b.get_job("job-1")["status"] == "done"
    assert b.get_job("job-1")["meta"]["steps"] == 20
    assert b.artifacts("job-1")[0]["path"] == str(path)
    assert remote.restore_file(path, media_root)
    assert path.read_bytes() == content

    s3.fail_write = True
    fails(lambda: b.create_job("lost", "image", "anything", {}))
    s3.fail_write = False

    remote.remove_file(path, media_root)
    assert not any(key.startswith(remote._object_prefix("job-one/video.mp4")) for key in s3.objects)

print("Neon Object Storage private bucket and recovery tests passed.")

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    prompt TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    kernel_ref TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    meta_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id)
);
CREATE INDEX IF NOT EXISTS idx_jobs_updated ON jobs(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_artifacts_job ON artifacts(job_id, created_at);
"""

class JobDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(SCHEMA)

    def _conn(self):
        con = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    def create_job(self, job_id: str, task: str, prompt: str, meta: dict[str, Any]) -> None:
        now = time.time()
        with _LOCK, self._conn() as con:
            con.execute(
                "INSERT INTO jobs(id,task,prompt,status,created_at,updated_at,meta_json) VALUES(?,?,?,?,?,?,?)",
                (job_id, task, prompt or "", "queued", now, now, json.dumps(meta, ensure_ascii=False)),
            )
            con.commit()

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {"status", "kernel_ref", "error", "meta_json", "prompt", "task"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        updates["updated_at"] = time.time()
        if "meta_json" in updates and not isinstance(updates["meta_json"], str):
            updates["meta_json"] = json.dumps(updates["meta_json"], ensure_ascii=False)
        sql = ", ".join(f"{k}=?" for k in updates)
        with _LOCK, self._conn() as con:
            con.execute(f"UPDATE jobs SET {sql} WHERE id=?", (*updates.values(), job_id))
            con.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with _LOCK, self._conn() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["meta"] = json.loads(item.pop("meta_json", "{}"))
        except Exception:
            item["meta"] = {}
        return item

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with _LOCK, self._conn() as con:
            rows = con.execute("SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def add_artifact(self, job_id: str, path: str, kind: str) -> None:
        with _LOCK, self._conn() as con:
            con.execute(
                "INSERT INTO artifacts(job_id,path,kind,created_at) VALUES(?,?,?,?)",
                (job_id, path, kind, time.time()),
            )
            con.commit()

    def artifacts(self, job_id: str) -> list[dict[str, Any]]:
        with _LOCK, self._conn() as con:
            rows = con.execute(
                "SELECT * FROM artifacts WHERE job_id=? ORDER BY created_at,id", (job_id,)
            ).fetchall()
        return [dict(r) for r in rows]

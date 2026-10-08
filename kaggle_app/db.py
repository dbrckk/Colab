from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .remote_storage import RemoteStorage

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

def _decode_job_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    try:
        item["meta"] = json.loads(item.get("meta_json") or "{}")
    except Exception:
        item["meta"] = {}
    return item

class JobDB:
    def __init__(self, path: Path, remote_store: RemoteStorage | None = None):
        self.path = Path(path)
        self.remote_store = remote_store
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.recovered_corrupt_path: Path | None = None
        if self.remote_store is not None:
            # Restore BEFORE schema migrations and before any job is resumed.
            self.remote_store.restore_database(self.path)
        self._initialize_database()
        self._persist()

    def _persist(self) -> None:
        if self.remote_store is not None:
            self.remote_store.save_database(self.path)

    def _initialize_database(self) -> None:
        try:
            with self._conn() as con:
                row = con.execute("PRAGMA integrity_check").fetchone()
                integrity = str(row[0]).lower() if row else "unknown"
                if integrity != "ok":
                    raise sqlite3.DatabaseError(f"integrity_check={integrity}")
                con.executescript(SCHEMA)
                self._migrate_artifact_uniqueness(con)
        except sqlite3.DatabaseError:
            if self.path.exists():
                stamp = time.strftime("%Y%m%d-%H%M%S")
                backup = self.path.with_name(self.path.name + f".corrupt-{stamp}")
                self.path.replace(backup)
                self.recovered_corrupt_path = backup
            with self._conn() as con:
                con.executescript(SCHEMA)
                self._migrate_artifact_uniqueness(con)

    def _migrate_artifact_uniqueness(self, con: sqlite3.Connection) -> None:
        # Older databases allowed duplicate artifact rows if the controller
        # stopped after copying outputs but before marking the job complete.
        con.execute(
            """
            DELETE FROM artifacts
            WHERE id NOT IN (
                SELECT MIN(id) FROM artifacts GROUP BY job_id, path
            )
            """
        )
        con.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_job_path "
            "ON artifacts(job_id, path)"
        )
        con.commit()

    def _conn(self):
        con = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        con.row_factory = sqlite3.Row
        # Conservative settings for mounted/persistent filesystems such as Google Drive.
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        con.execute("PRAGMA journal_mode=DELETE")
        con.execute("PRAGMA synchronous=FULL")
        return con

    def integrity_status(self) -> str:
        try:
            with self._conn() as con:
                row = con.execute("PRAGMA integrity_check").fetchone()
            result = str(row[0]) if row else "unknown"
            if self.recovered_corrupt_path:
                return f"{result} (ancienne base sauvegardée: {self.recovered_corrupt_path})"
            return result
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    def create_job(self, job_id: str, task: str, prompt: str, meta: dict[str, Any]) -> None:
        now = time.time()
        with _LOCK, self._conn() as con:
            con.execute(
                "INSERT INTO jobs(id,task,prompt,status,created_at,updated_at,meta_json) VALUES(?,?,?,?,?,?,?)",
                (job_id, task, prompt or "", "queued", now, now, json.dumps(meta, ensure_ascii=False)),
            )
            con.commit()
            self._persist()

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
            self._persist()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with _LOCK, self._conn() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        return _decode_job_row(row)

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with _LOCK, self._conn() as con:
            rows = con.execute("SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [_decode_job_row(r) for r in rows]

    def add_artifact(self, job_id: str, path: str, kind: str) -> None:
        with _LOCK, self._conn() as con:
            con.execute(
                "INSERT OR IGNORE INTO artifacts(job_id,path,kind,created_at) VALUES(?,?,?,?)",
                (job_id, path, kind, time.time()),
            )
            con.commit()
            self._persist()

    def artifacts(self, job_id: str) -> list[dict[str, Any]]:
        with _LOCK, self._conn() as con:
            rows = con.execute(
                "SELECT * FROM artifacts WHERE job_id=? ORDER BY created_at,id", (job_id,)
            ).fetchall()
        return [dict(r) for r in rows]


    def delete_job(self, job_id: str) -> None:
        with _LOCK, self._conn() as con:
            con.execute("DELETE FROM artifacts WHERE job_id=?", (job_id,))
            con.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            con.commit()
            self._persist()


    def recent_artifacts(self, kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = """
        SELECT artifacts.*, jobs.prompt, jobs.task, jobs.status
        FROM artifacts
        JOIN jobs ON jobs.id = artifacts.job_id
        """
        params: list[Any] = []
        if kind:
            sql += " WHERE artifacts.kind=?"
            params.append(kind)
        sql += " ORDER BY artifacts.created_at DESC, artifacts.id DESC LIMIT ?"
        params.append(int(limit))
        with _LOCK, self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


    def delete_artifacts_by_ids(self, ids: list[int]) -> int:
        clean = [int(x) for x in ids if int(x) > 0]
        if not clean:
            return 0
        placeholders = ",".join("?" for _ in clean)
        with _LOCK, self._conn() as con:
            cur = con.execute(
                f"DELETE FROM artifacts WHERE id IN ({placeholders})",
                clean,
            )
            con.commit()
            self._persist()
            return int(cur.rowcount or 0)

    def vacuum(self) -> None:
        with _LOCK, self._conn() as con:
            con.execute("VACUUM")
            self._persist()

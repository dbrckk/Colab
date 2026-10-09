from __future__ import annotations

"""Neon PostgreSQL adapter for the existing private, verified chunk store.

The controller uses this single-writer object store for SQLite snapshots and
media chunks. No database password or object is sent to the browser.
"""

import os
from urllib.parse import urlsplit

from .remote_storage import CHUNK_SIZE, MAX_DATABASE_BYTES, RemoteStorage, RemoteStorageError

DEFAULT_QUOTA = 512 * 1024 * 1024


class NeonStorage(RemoteStorage):
    backend_name = "Neon PostgreSQL"

    def __init__(
        self,
        database_url: str,
        bucket: str = "qwen-studio-private",
        max_bytes: int = DEFAULT_QUOTA,
    ):
        parsed = urlsplit(database_url)
        if (
            parsed.scheme not in {"postgresql", "postgres"}
            or not parsed.hostname
            or not parsed.username
            or not parsed.password
        ):
            raise RemoteStorageError("URL de connexion PostgreSQL Neon invalide.")
        if not bucket or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in bucket):
            raise RemoteStorageError("Nom de préfixe Neon invalide.")
        if max_bytes < MAX_DATABASE_BYTES:
            raise RemoteStorageError("Quota Neon trop faible pour les sauvegardes.")
        self.database_url = database_url
        self.bucket = bucket
        self.max_bytes = int(max_bytes)

    @classmethod
    def from_env(cls) -> NeonStorage | None:
        database_url = os.getenv("QWEN_NEON_DATABASE_URL", "").strip()
        if not database_url:
            return None
        quota = os.getenv("QWEN_NEON_MAX_STORAGE_BYTES", str(DEFAULT_QUOTA))
        try:
            maximum = int(quota)
        except ValueError:
            raise RemoteStorageError("QWEN_NEON_MAX_STORAGE_BYTES doit être un entier.") from None
        return cls(
            database_url,
            bucket=os.getenv("QWEN_NEON_BUCKET", "qwen-studio-private"),
            max_bytes=maximum,
        )

    def _connect(self):
        # Import only when enabled: normal Colab and Supabase installations
        # do not need psycopg.
        import psycopg

        return psycopg.connect(
            self.database_url,
            connect_timeout=15,
            sslmode="require",
        )

    def _key(self, key: str) -> str:
        if not key or key.startswith("/") or ".." in key.split("/"):
            raise RemoteStorageError("Clé de stockage Neon invalide.")
        return f"{self.bucket}/{key}"

    def ensure_private_bucket(self) -> None:
        # Neon exposes PostgreSQL connections only; there is no public bucket.
        # The dedicated database login has exclusive access to the table.
        try:
            with self._connect() as con:
                with con.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS qwen_private_objects (
                            object_key TEXT PRIMARY KEY,
                            data BYTEA NOT NULL,
                            mime TEXT NOT NULL,
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                        """
                    )
        except Exception:
            raise RemoteStorageError(
                "Initialisation du stockage privé Neon impossible."
            ) from None

    def object_get(self, key: str) -> bytes | None:
        try:
            with self._connect() as con:
                with con.cursor() as cur:
                    cur.execute(
                        "SELECT data FROM qwen_private_objects WHERE object_key = %s",
                        (self._key(key),),
                    )
                    row = cur.fetchone()
            return bytes(row[0]) if row else None
        except RemoteStorageError:
            raise
        except Exception:
            raise RemoteStorageError("Lecture du stockage Neon impossible.") from None

    def object_put(self, key: str, content: bytes, mime: str) -> None:
        if len(content) > max(CHUNK_SIZE, MAX_DATABASE_BYTES):
            raise RemoteStorageError("Objet trop volumineux pour Neon.")
        try:
            with self._connect() as con:
                with con.cursor() as cur:
                    # This controller runs as one Render instance. The
                    # transaction-level advisory lock also guards concurrent
                    # local controller threads against quota races.
                    cur.execute("SELECT pg_advisory_xact_lock(82642019)")
                    cur.execute(
                        "SELECT COALESCE(SUM(octet_length(data)), 0) "
                        "FROM qwen_private_objects"
                    )
                    used = int(cur.fetchone()[0])
                    cur.execute(
                        "SELECT octet_length(data) FROM qwen_private_objects "
                        "WHERE object_key = %s",
                        (self._key(key),),
                    )
                    old = cur.fetchone()
                    previous = int(old[0]) if old else 0
                    if used - previous + len(content) > self.max_bytes:
                        raise RemoteStorageError(
                            "Quota privé Neon atteint. Télécharge ou supprime "
                            "des médias, ou configure un stockage objet externe."
                        )
                    cur.execute(
                        """
                        INSERT INTO qwen_private_objects (object_key, data, mime)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (object_key) DO UPDATE
                        SET data = EXCLUDED.data,
                            mime = EXCLUDED.mime,
                            updated_at = now()
                        """,
                        (self._key(key), content, mime),
                    )
        except RemoteStorageError:
            raise
        except Exception:
            # Never leak a PostgreSQL DSN or server query through Gradio/errors.
            raise RemoteStorageError("Écriture du stockage Neon impossible.") from None

    def object_remove(self, keys: list[str]) -> None:
        if not keys:
            return
        resolved = [self._key(key) for key in keys]
        try:
            with self._connect() as con:
                with con.cursor() as cur:
                    cur.execute(
                        "DELETE FROM qwen_private_objects "
                        "WHERE object_key = ANY(%s)",
                        (resolved,),
                    )
        except RemoteStorageError:
            raise
        except Exception:
            raise RemoteStorageError("Suppression du stockage Neon impossible.") from None

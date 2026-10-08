from __future__ import annotations

"""Private Supabase Storage persistence for single-instance hosted controllers.

No service credential ever leaves the server. Uploaded content is stored as
content-addressed chunks. A complete, checksum-verified manifest is published
only after every chunk was uploaded successfully.
"""

import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import requests

CHUNK_SIZE = 8 * 1024 * 1024
DB_OBJECT = "state/jobs.sqlite3.gz"
MAX_DATABASE_BYTES = 64 * 1024 * 1024


class RemoteStorageError(RuntimeError):
    pass


class RemoteStorage:
    def __init__(self, url: str, service_key: str, bucket: str = "qwen-studio-private"):
        url = url.rstrip("/")
        if not url.startswith("https://") or not service_key.strip():
            raise RemoteStorageError("URL HTTPS Supabase et clé serveur obligatoires.")
        if not bucket or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in bucket):
            raise RemoteStorageError("Nom de bucket Supabase invalide.")
        self.url = url
        self.service_key = service_key.strip()
        self.bucket = bucket

    @classmethod
    def from_env(cls) -> RemoteStorage | None:
        url = os.getenv("QWEN_SUPABASE_URL", "").strip()
        key = os.getenv("QWEN_SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if not url and not key:
            return None
        if not url or not key:
            raise RemoteStorageError(
                "Configuration Supabase incomplète : QWEN_SUPABASE_URL et "
                "QWEN_SUPABASE_SERVICE_ROLE_KEY sont nécessaires."
            )
        return cls(url, key, os.getenv("QWEN_SUPABASE_BUCKET", "qwen-studio-private"))

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        data: bytes | None = None,
        content_type: str = "application/octet-stream",
        headers: dict[str, str] | None = None,
        missing_ok: bool = False,
    ) -> bytes | None:
        base_headers = {
            "apikey": self.service_key,
            "Cache-Control": "no-cache",
        }
        # Modern sb_secret_* API keys are not JWTs. Supabase expects them
        # in apikey, whereas legacy service_role JWTs can be used as Bearer.
        if not self.service_key.startswith("sb_secret_"):
            base_headers["Authorization"] = f"Bearer {self.service_key}"
        if data is not None:
            base_headers["Content-Type"] = content_type
        if headers:
            base_headers.update(headers)
        # Service key is in headers only; never in a URL or exception message.
        for attempt in range(3):
            try:
                response = requests.request(
                    method, f"{self.url}/storage/v1/{endpoint}",
                    headers=base_headers, data=data, timeout=(10, 90),
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RemoteStorageError(
                    f"Stockage distant inaccessible ({type(exc).__name__})."
                ) from None

            if response.status_code == 404 and missing_ok:
                return None
            if 200 <= response.status_code < 300:
                return response.content
            if response.status_code in (408, 429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RemoteStorageError(
                f"Supabase Storage HTTP {response.status_code} sur {method} "
                f"(vérifier l'état du projet, le bucket et les permissions)."
            )
        raise RemoteStorageError("Échec inattendu de la requête Supabase.")

    def ensure_private_bucket(self) -> None:
        bucket = quote(self.bucket, safe="")
        response = self._request("GET", f"bucket/{bucket}", missing_ok=True)
        if response is None:
            payload = json.dumps({
                "id": self.bucket, "name": self.bucket, "public": False
            }).encode("utf-8")
            self._request(
                "POST", "bucket", data=payload, content_type="application/json"
            )
            response = self._request("GET", f"bucket/{bucket}")
        try:
            info = json.loads(response or b"{}")
        except (ValueError, TypeError):
            raise RemoteStorageError("Réponse bucket Supabase invalide.") from None
        if info.get("public") is not False:
            raise RemoteStorageError(
                "Le bucket Supabase existe mais n'est pas privé : arrêt de sécurité."
            )

    def object_get(self, key: str) -> bytes | None:
        path = quote(f"{self.bucket}/{key}", safe="/")
        return self._request("GET", f"object/authenticated/{path}", missing_ok=True)

    def object_put(self, key: str, content: bytes, mime: str) -> None:
        path = quote(f"{self.bucket}/{key}", safe="/")
        self._request(
            "POST", f"object/{path}", data=content, content_type=mime,
            headers={"x-upsert": "true"},
        )

    def object_remove(self, keys: list[str]) -> None:
        if not keys:
            return
        content = json.dumps({"prefixes": keys}).encode("utf-8")
        self._request(
            "DELETE", f"object/{quote(self.bucket, safe='')}",
            data=content, content_type="application/json",
        )

    @staticmethod
    def _relative(path: Path, root: Path) -> str:
        # resolve() also blocks symlinks pointing outside the storage root.
        try:
            relative = path.resolve().relative_to(root.resolve())
        except ValueError:
            raise RemoteStorageError("Fichier hors du répertoire autorisé.") from None
        if not relative.parts or any(p.startswith(".") for p in relative.parts):
            raise RemoteStorageError("Chemin de média distant invalide.")
        return relative.as_posix()

    @staticmethod
    def _object_prefix(relative: str) -> str:
        return "objects/" + hashlib.sha256(relative.encode("utf-8")).hexdigest()

    def _manifest(self, relative: str) -> dict | None:
        raw = self.object_get(f"{self._object_prefix(relative)}/manifest.json")
        if raw is None:
            return None
        try:
            manifest = json.loads(raw)
            parts = manifest["parts"]
            if (
                manifest["version"] != 1
                or manifest["relative"] != relative
                or not isinstance(parts, list)
                or not isinstance(manifest["size"], int)
                or manifest["size"] <= 0
                or len(manifest["sha256"]) != 64
            ):
                raise ValueError("invalid manifest")
            if sum(int(p["size"]) for p in parts) != manifest["size"]:
                raise ValueError("invalid manifest sizes")
            for part in parts:
                if (
                    not isinstance(part["size"], int) or not 0 < part["size"] <= CHUNK_SIZE
                    or len(part["sha256"]) != 64
                    or not str(part["key"]).startswith(self._object_prefix(relative) + "/versions/")
                ):
                    raise ValueError("invalid part")
            return manifest
        except (KeyError, TypeError, ValueError) as exc:
            raise RemoteStorageError("Manifeste distant corrompu.") from exc

    def save_file(self, path: Path, root: Path) -> None:
        relative = self._relative(path, root)
        if not path.is_file():
            raise FileNotFoundError(path)
        start = path.stat()
        if not start.st_size:
            raise RemoteStorageError("Un fichier vide ne peut pas être sauvegardé.")
        sha = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(CHUNK_SIZE), b""):
                sha.update(block)
        digest = sha.hexdigest()
        old = self._manifest(relative)
        if old and old["sha256"] == digest and old["size"] == start.st_size:
            return

        prefix = self._object_prefix(relative)
        parts: list[dict] = []
        with path.open("rb") as source:
            index = 0
            while block := source.read(CHUNK_SIZE):
                part_key = f"{prefix}/versions/{digest}/{index:06d}.bin"
                self.object_put(part_key, block, "application/octet-stream")
                parts.append({
                    "key": part_key, "size": len(block),
                    "sha256": hashlib.sha256(block).hexdigest(),
                })
                index += 1

        end = path.stat()
        if start.st_size != end.st_size or start.st_mtime_ns != end.st_mtime_ns:
            raise RemoteStorageError("Média modifié pendant son transfert.")
        manifest = {
            "version": 1, "relative": relative,
            "size": start.st_size, "sha256": digest, "parts": parts,
        }
        self.object_put(
            f"{prefix}/manifest.json",
            json.dumps(manifest, separators=(",", ":")).encode("utf-8"),
            "application/json",
        )

    def restore_file(self, path: Path, root: Path) -> bool:
        relative = self._relative(path, root)
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return True
        manifest = self._manifest(relative)
        if manifest is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".restore-", dir=str(path.parent))
        os.close(fd)
        temp = Path(name)
        try:
            digest = hashlib.sha256()
            size = 0
            with temp.open("wb") as dest:
                for item in manifest["parts"]:
                    chunk = self.object_get(item["key"])
                    if (
                        chunk is None or len(chunk) != item["size"]
                        or hashlib.sha256(chunk).hexdigest() != item["sha256"]
                    ):
                        raise RemoteStorageError("Chunk distant manquant ou altéré.")
                    dest.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != manifest["size"] or digest.hexdigest() != manifest["sha256"]:
                raise RemoteStorageError("Somme SHA-256 distante invalide.")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return True

    def remove_file(self, path: Path, root: Path) -> None:
        relative = self._relative(path, root)
        manifest = self._manifest(relative)
        if not manifest:
            return
        keys = [x["key"] for x in manifest["parts"]]
        keys.append(f"{self._object_prefix(relative)}/manifest.json")
        # Supabase remove() supports batches up to 1000 paths.
        for start in range(0, len(keys), 1000):
            self.object_remove(keys[start:start + 1000])

    def restore_database(self, target: Path) -> bool:
        packed = self.object_get(DB_OBJECT)
        if packed is None:
            return False
        try:
            import io
            with gzip.GzipFile(fileobj=io.BytesIO(packed), mode="rb") as gz:
                raw = gz.read(MAX_DATABASE_BYTES + 1)
                if len(raw) > MAX_DATABASE_BYTES:
                    raise RemoteStorageError("Base distante trop volumineuse.")
        except (OSError, EOFError) as exc:
            raise RemoteStorageError("Sauvegarde SQLite distante illisible.") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".restore-db-", dir=str(target.parent))
        os.close(fd)
        tmp = Path(name)
        try:
            tmp.write_bytes(raw)
            with sqlite3.connect(tmp) as con:
                check = con.execute("PRAGMA integrity_check").fetchone()
                if check is None or check[0] != "ok":
                    raise RemoteStorageError("Intégrité SQLite distante invalide.")
            os.replace(tmp, target)
        except sqlite3.DatabaseError as exc:
            raise RemoteStorageError("Base SQLite distante corrompue.") from exc
        finally:
            tmp.unlink(missing_ok=True)
        return True

    def save_database(self, target: Path) -> None:
        with sqlite3.connect(target, timeout=30) as con:
            raw = con.serialize()
        if len(raw) > MAX_DATABASE_BYTES:
            raise RemoteStorageError("Base SQLite trop volumineuse pour la sauvegarde.")
        self.object_put(DB_OBJECT, gzip.compress(raw, compresslevel=6), "application/gzip")

from __future__ import annotations

"""Private Neon Object Storage using the verified RemoteStorage protocol.

S3 credentials are scoped to this Neon branch and remain on the server.
The parent class provides atomic manifests, chunk checksums, and SQLite
snapshot / media cold-start restore.
"""

import os
from urllib.parse import urlsplit

from .remote_storage import RemoteStorage, RemoteStorageError


class NeonObjectStorage(RemoteStorage):
    backend_name = "Neon Object Storage"

    def __init__(
        self,
        endpoint: str,
        access_key_id: str,
        secret_access_key: str,
        region: str,
        bucket: str = "qwen-studio-private",
        *,
        client=None,
    ):
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.endswith(".neon.tech")
            or parsed.username or parsed.password or parsed.query or parsed.fragment
        ):
            raise RemoteStorageError("Endpoint HTTPS Neon Storage invalide.")
        if not access_key_id.startswith("nak_") or not secret_access_key.startswith("nsk_"):
            raise RemoteStorageError("Identifiants S3 Neon invalides.")
        if not region or any(x not in "abcdefghijklmnopqrstuvwxyz0123456789-" for x in region):
            raise RemoteStorageError("Région S3 invalide.")
        if not bucket or any(x not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for x in bucket):
            raise RemoteStorageError("Nom de bucket Neon invalide.")

        self.bucket = bucket
        self.url = endpoint.rstrip("/")
        self.region = region
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "s3",
                endpoint_url=self.url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    connect_timeout=10,
                    read_timeout=90,
                    retries={"mode": "standard", "max_attempts": 3},
                ),
            )
        self.client = client

    @classmethod
    def from_env(cls) -> NeonObjectStorage | None:
        env = {
            "endpoint": os.getenv("QWEN_NEON_S3_ENDPOINT", "").strip(),
            "access_key_id": os.getenv("QWEN_NEON_S3_ACCESS_KEY_ID", "").strip(),
            "secret_access_key": os.getenv("QWEN_NEON_S3_SECRET_ACCESS_KEY", "").strip(),
            "region": os.getenv("QWEN_NEON_S3_REGION", "").strip(),
        }
        if not any(env.values()):
            return None
        if not all(env.values()):
            raise RemoteStorageError(
                "Configuration Neon S3 incomplète : endpoint, région et deux "
                "identifiants sont obligatoires."
            )
        return cls(
            **env,
            bucket=os.getenv("QWEN_NEON_S3_BUCKET", "qwen-studio-private"),
        )

    def ensure_private_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            acl = self.client.get_bucket_acl(Bucket=self.bucket)
            grants = acl.get("Grants") or []
            for grant in grants:
                grantee = grant.get("Grantee") or {}
                uri = str(grantee.get("URI") or "")
                if uri.endswith("/AllUsers") or uri.endswith("/AuthenticatedUsers"):
                    raise RemoteStorageError(
                        "Le bucket Neon n'est pas privé : démarrage interrompu."
                    )
        except RemoteStorageError:
            raise
        except Exception:
            raise RemoteStorageError(
                "Bucket Neon inaccessible ou permissions insuffisantes."
            ) from None

    def object_get(self, key: str) -> bytes | None:
        try:
            result = self.client.get_object(Bucket=self.bucket, Key=key)
            body = result["Body"]
            try:
                return body.read()
            finally:
                body.close()
        except Exception as exc:
            # A missing object is normal on the initial install. Other S3
            # failures must never be mistaken for an empty storage.
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise RemoteStorageError(
                "Lecture Neon Object Storage impossible."
            ) from None

    def object_put(self, key: str, content: bytes, mime: str) -> None:
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content,
                ContentType=mime,
            )
        except Exception:
            raise RemoteStorageError(
                "Écriture Neon Object Storage impossible."
            ) from None

    def object_remove(self, keys: list[str]) -> None:
        for start in range(0, len(keys), 1000):
            group = keys[start:start + 1000]
            try:
                result = self.client.delete_objects(
                    Bucket=self.bucket,
                    Delete={
                        "Objects": [{"Key": key} for key in group],
                        "Quiet": True,
                    },
                )
                if result.get("Errors"):
                    raise RemoteStorageError(
                        "Suppression partielle d'objets Neon."
                    )
            except RemoteStorageError:
                raise
            except Exception:
                raise RemoteStorageError(
                    "Suppression Neon Object Storage impossible."
                ) from None

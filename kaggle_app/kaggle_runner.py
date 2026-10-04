from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import SETTINGS, Settings
from .db import JobDB
from .kaggle_auth import (
    apply_credential_env,
    auth_cache_valid,
    credential_fingerprint,
    credential_save_message,
    credentials_ready as auth_credentials_ready,
    restore_credential_env,
    snapshot_credential_env,
    validate_credential_fields,
    write_credential_env,
)
from .kaggle_protocol import (
    is_auth_cli_error,
    is_transient_cli_error,
    parse_dataset_status,
    parse_kernel_status,
    remote_kernel_missing,
)
from .kaggle_cleanup import cleanup_report, expired_exports, orphan_input_dirs
from .kaggle_dashboard import dashboard_summary_text
from .kaggle_dataset import write_dataset_bundle
from .kaggle_health import recovery_diagnostic_lines
from .kaggle_input_store import (
    ensure_input_store,
    hash_file as input_store_hash_file,
    persist_input as persist_input_file,
    unreferenced_store_files,
)
from .kaggle_inputs import validate_submission
from .kaggle_jobs import (
    build_job_config,
    dataset_ref as build_dataset_ref,
    kernel_ref as build_kernel_ref,
    needs_dataset,
    slugify,
)
from .kaggle_kernel import write_kernel_bundle
from .kaggle_monitor import wait_for_kernel
from .kaggle_outputs import validate_downloaded_outputs
from .kaggle_reconcile import classify_missing_local_outputs, has_valid_local_media
from .kaggle_recovery import (
    auto_recovery_attempts,
    classify_remote_recovery_failure,
    has_recoverable_outputs,
    keep_remote_kernel_for_retry,
    preserve_kernel_after_execute_failure,
    recoverable_output_candidate,
    recovery_marked,
    remote_kernel_is_preserved,
    startup_recovery_action,
)
from .storage import import_outputs, scan_outputs

# Backward-compatible aliases kept for existing tests/importers while the
# implementation lives in kaggle_protocol.py.
_is_auth_cli_error = is_auth_cli_error
_is_transient_cli_error = is_transient_cli_error

class JobCancelled(RuntimeError):
    pass

class RemoteKernelFailed(RuntimeError):
    """Kaggle explicitly reported a terminal kernel failure."""
    pass

class KaggleController:
    def __init__(self, settings: Settings = SETTINGS):
        self.settings = settings
        self.db = JobDB(settings.db_path)
        # Kaggle GPU sessions are intentionally serialized.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kaggle-job")
        self._futures: dict[str, Any] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()
        self._storage_cache = {"ts": 0.0, "bytes": 0}
        self._auth_cache = {"fingerprint": "", "ts": 0.0}
        self._recover_persisted_jobs()

    def _recover_persisted_jobs(self) -> None:
        credentials_ready = self.credentials_ready()
        for row in self.db.list_jobs(200):
            job_id = row.get("id")
            if not job_id:
                continue
            action = startup_recovery_action(row, credentials_ready)
            kernel_ref = row.get("kernel_ref") or ""
            meta = row.get("meta") or {}

            if action == "resume_local":
                dataset_ref = meta.get("dataset_ref") or ""
                if dataset_ref:
                    try:
                        self._cleanup_remote_refs("", dataset_ref)
                    except Exception:
                        pass
                self.db.update_job(
                    job_id,
                    status="queued",
                    error="",
                    meta_json={**meta, "dataset_ref": ""},
                )
                with self._lock:
                    self._futures[job_id] = self.executor.submit(self._execute, job_id)
            elif action == "wait_local_auth":
                self.db.update_job(
                    job_id,
                    status="waiting_auth",
                    error="En attente des identifiants Kaggle pour reprendre la préparation locale.",
                )
            elif action == "resume_remote":
                if row.get("status") == "waiting_auth":
                    self.db.update_job(job_id, status="recovering", error="")
                with self._lock:
                    self._futures[job_id] = self.executor.submit(self._recover_remote_job, job_id)
            elif action == "wait_remote_auth":
                self.db.update_job(
                    job_id,
                    status="waiting_auth",
                    error="En attente des identifiants Kaggle pour reprendre le kernel distant existant.",
                )
            elif action == "resume_recoverable":
                attempts = auto_recovery_attempts(row)
                self.db.update_job(
                    job_id,
                    status="recovering",
                    error="",
                    meta_json={
                        **meta,
                        "auto_recovery_attempts": attempts + 1,
                        "last_auto_recovery_at": time.time(),
                    },
                )
                with self._lock:
                    self._futures[job_id] = self.executor.submit(self._recover_remote_job, job_id)

    def _media_integrity_ok(self, path: Path, kind: str) -> bool:
        try:
            if not path.exists() or path.stat().st_size <= 0:
                return False
            if kind == "image":
                try:
                    from PIL import Image
                    with Image.open(path) as img:
                        img.verify()
                    return True
                except Exception:
                    return False
            if kind == "video":
                ffprobe = shutil.which("ffprobe")
                if not ffprobe:
                    return path.stat().st_size > 1024
                p = subprocess.run(
                    [
                        ffprobe, "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=codec_type",
                        "-of", "default=nokey=1:noprint_wrappers=1",
                        str(path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return p.returncode == 0 and "video" in (p.stdout or "").lower()
            return True
        except Exception:
            return False

    def _validate_downloaded_outputs(
        self,
        job: dict[str, Any],
        artifacts: list[tuple[Path, str]],
    ) -> dict[str, Any]:
        return validate_downloaded_outputs(
            job,
            artifacts,
            hash_file=self._hash_file,
            media_integrity_ok=self._media_integrity_ok,
        )

    @staticmethod
    def _remote_kernel_missing(exc: Exception) -> bool:
        return remote_kernel_missing(str(exc))

    def _recover_remote_job(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if not job:
            return
        kernel_ref = job.get("kernel_ref") or ""
        meta = job.get("meta") or {}
        dataset_ref = meta.get("dataset_ref") or ""
        if not kernel_ref:
            self.db.update_job(
                job_id,
                status="interrupted",
                error="Kernel Kaggle introuvable pour la reprise.",
            )
            return

        try:
            self.validate_current_credentials()
            self.db.update_job(job_id, status="recovering")
            wait_for_kernel(
                kernel_ref,
                kernel_timeout=self.settings.kernel_timeout,
                poll_seconds=self.settings.poll_seconds,
                check_cancelled=lambda: self._check_cancelled(job_id),
                kernel_status=self._kernel_status,
                on_status=lambda state, raw: self.db.update_job(
                    job_id,
                    status=("recovering" if state in {"queued", "running"} else state),
                    meta_json={
                        **meta,
                        "dataset_ref": dataset_ref,
                        "kaggle_status": raw[-1500:],
                    },
                ),
                fetch_logs=lambda ref: self._run(["kernels", "logs", ref], timeout=180),
                failure_factory=lambda logs: RemoteKernelFailed(
                    "Le job Kaggle récupéré a échoué:\n" + logs[-7000:]
                ),
                timeout_message="Délai maximal dépassé pendant la récupération Kaggle.",
                now=time.time,
                sleep=time.sleep,
            )

            with tempfile.TemporaryDirectory(prefix=f"qwen-kaggle-recover-{job_id}-") as td:
                download = Path(td) / "download"
                self.db.update_job(job_id, status="downloading")
                _, result_manifest = self._download_validated_outputs(
                    job, kernel_ref, download
                )
                self._commit_downloaded_outputs(
                    job,
                    download,
                    result_manifest,
                )
        except JobCancelled:
            self.db.update_job(job_id, status="cancelled", error="")
        except Exception as exc:
            if self.is_cancelled(job_id):
                self.db.update_job(job_id, status="cancelled", error="")
            else:
                current = self.db.get_job(job_id) or job
                current_meta = current.get("meta") or {}
                missing_remote = self._remote_kernel_missing(exc)
                remote_failed = isinstance(exc, RemoteKernelFailed)
                auth_required = is_auth_cli_error(str(exc))
                (
                    next_status,
                    recover_outputs_available,
                    failed_phase,
                    remote_failure_confirmed,
                ) = classify_remote_recovery_failure(
                    auth_required=auth_required,
                    missing_remote=missing_remote,
                    remote_failed=remote_failed,
                )
                self.db.update_job(
                    job_id,
                    status=next_status,
                    error=(
                        "Authentification Kaggle requise pour reprendre le kernel distant existant."
                        if auth_required
                        else (
                            "Kernel Kaggle supprimé ou introuvable; utilise Relancer pour recalculer."
                            if missing_remote
                            else f"Recovery {type(exc).__name__}: {exc}"
                        )
                    ),
                    meta_json={
                        **current_meta,
                        "recover_outputs_available": recover_outputs_available,
                        "failed_phase": failed_phase,
                        "remote_failure_confirmed": remote_failure_confirmed,
                    },
                )
        finally:
            current = self.db.get_job(job_id) or {}
            current_meta = current.get("meta") or {}
            keep_for_retry = keep_remote_kernel_for_retry(
                str(current.get("status") or ""),
                current_meta,
                kernel_ref,
            )
            if keep_for_retry:
                # The completed kernel is the recovery source. Keep it across
                # controller restarts and repeated download attempts.
                self._cleanup_remote_refs("", dataset_ref)
            else:
                self._cleanup_remote_refs(kernel_ref, dataset_ref)
            with self._lock:
                self._cancelled.discard(job_id)

    def resume_waiting_jobs(self) -> int:
        if not self.credentials_ready():
            return 0
        try:
            self.validate_current_credentials()
        except Exception:
            return 0
        resumed = 0
        for row in self.db.list_jobs(500):
            if row.get("status") != "waiting_auth":
                continue
            job_id = row.get("id")
            if not job_id:
                continue
            with self._lock:
                future = self._futures.get(job_id)
                if future is not None and not future.done():
                    continue
                kernel_ref = row.get("kernel_ref") or ""
                if kernel_ref:
                    self.db.update_job(job_id, status="recovering", error="")
                    self._futures[job_id] = self.executor.submit(
                        self._recover_remote_job, job_id
                    )
                else:
                    meta = row.get("meta") or {}
                    dataset_ref = meta.get("dataset_ref") or ""
                    if dataset_ref:
                        self._cleanup_remote_refs("", dataset_ref)
                    self.db.update_job(
                        job_id,
                        status="queued",
                        error="",
                        meta_json={**meta, "dataset_ref": ""},
                    )
                    self._futures[job_id] = self.executor.submit(self._execute, job_id)
                resumed += 1
        return resumed

    def resume_recoverable_outputs(self) -> int:
        if not self.credentials_ready():
            return 0
        try:
            self.validate_current_credentials()
        except Exception:
            return 0
        resumed = 0
        for row in self.db.list_jobs(500):
            if not recoverable_output_candidate(row):
                continue
            kernel_ref = row.get("kernel_ref") or ""
            meta = row.get("meta") or {}
            attempts = auto_recovery_attempts(row)
            job_id = row.get("id")
            if not job_id:
                continue
            with self._lock:
                future = self._futures.get(job_id)
                if future is not None and not future.done():
                    continue
                self.db.update_job(
                    job_id,
                    status="recovering",
                    error="",
                    meta_json={
                        **meta,
                        "auto_recovery_attempts": attempts + 1,
                        "last_auto_recovery_at": time.time(),
                    },
                )
                self._futures[job_id] = self.executor.submit(self._recover_remote_job, job_id)
                resumed += 1
        return resumed

    def credentials_ready(self) -> bool:
        return auth_credentials_ready(
            self.settings.kaggle_username,
            self.settings.kaggle_api_token,
            self.settings.kaggle_key,
        )

    def _credential_fingerprint(self) -> str:
        return credential_fingerprint(
            self.settings.kaggle_username,
            self.settings.kaggle_api_token or self.settings.kaggle_key,
        )

    def credentials_recently_validated(self, max_age_seconds: int = 300) -> bool:
        return auth_cache_valid(
            self._auth_cache,
            self._credential_fingerprint(),
            now=time.time(),
            max_age_seconds=max_age_seconds,
        )

    def validate_current_credentials(self, max_age_seconds: int = 300) -> None:
        if not self.credentials_ready():
            raise RuntimeError(
                "Configure d'abord KAGGLE_USERNAME et KAGGLE_API_TOKEN "
                "(ou KAGGLE_KEY legacy)."
            )
        fingerprint = self._credential_fingerprint()
        now = time.time()
        if (
            fingerprint
            and self._auth_cache.get("fingerprint") == fingerprint
            and now - float(self._auth_cache.get("ts") or 0) <= max(0, int(max_age_seconds))
        ):
            return
        try:
            self._run(["kernels", "list", "-m", "-p", "1"], timeout=120)
        except Exception:
            # A forced/expired validation must never leave a stale "ready"
            # indicator for the same credentials.
            if self._auth_cache.get("fingerprint") == fingerprint:
                self._auth_cache = {"fingerprint": "", "ts": 0.0}
            raise
        self._auth_cache = {"fingerprint": fingerprint, "ts": time.time()}

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.settings.kaggle_username:
            env["KAGGLE_USERNAME"] = self.settings.kaggle_username
        if self.settings.kaggle_api_token:
            env["KAGGLE_API_TOKEN"] = self.settings.kaggle_api_token
        if self.settings.kaggle_key:
            env["KAGGLE_KEY"] = self.settings.kaggle_key
        return env

    def ensure_cli(self) -> str:
        exe = shutil.which("kaggle")
        current = (0, 0, 0)
        try:
            raw = version("kaggle")
            nums = [int(x) for x in re.findall(r"\d+", raw)[:3]]
            current = tuple((nums + [0, 0, 0])[:3])
        except (PackageNotFoundError, ValueError):
            pass
        if exe and current >= (2, 2, 3):
            return exe

        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--disable-pip-version-check", "--quiet", "--upgrade",
                "kaggle>=2.2.3",
            ],
            check=True,
            timeout=300,
        )
        exe = shutil.which("kaggle")
        if not exe:
            raise RuntimeError("Le CLI Kaggle n'a pas été trouvé après installation.")
        return exe

    def _run(
        self,
        args: list[str],
        timeout: int | None = None,
        retries: int | None = None,
    ) -> str:
        exe = self.ensure_cli()
        attempts = max(
            1,
            int(self.settings.cli_retries if retries is None else retries),
        )
        last_output = ""
        for attempt in range(1, attempts + 1):
            try:
                p = subprocess.run(
                    [exe, *args],
                    capture_output=True,
                    text=True,
                    env=self._env(),
                    timeout=timeout,
                )
                output = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
                last_output = output
                if p.returncode == 0:
                    return output
                if attempt >= attempts or not is_transient_cli_error(output):
                    raise RuntimeError(output[-7000:] or f"Kaggle CLI exit={p.returncode}")
            except subprocess.TimeoutExpired as exc:
                last_output = f"Kaggle CLI timeout: {exc}"
                if attempt >= attempts:
                    raise RuntimeError(last_output) from exc

            # 2s, 4s, 8s... capped to avoid freezing the controller too long.
            time.sleep(min(12, 2 ** attempt))

        raise RuntimeError(last_output[-7000:] or "Échec Kaggle CLI après plusieurs tentatives.")


    def save_credentials(
        self,
        username: str,
        api_token: str = "",
        legacy_key: str = "",
        persist: bool = True,
    ) -> str:
        username = (username or "").strip()
        api_token = (api_token or "").strip()
        legacy_key = (legacy_key or "").strip()
        # Password fields intentionally stay blank in the UI. If Colab Secrets
        # or the current process already provided valid credentials, allow the
        # user to test/persist them without copying the secret back into the UI.
        current_username = self.settings.kaggle_username
        if not api_token and not legacy_key and username == current_username:
            api_token = self.settings.kaggle_api_token
            legacy_key = "" if api_token else self.settings.kaggle_key

        validate_credential_fields(username, api_token, legacy_key)

        previous_env = snapshot_credential_env()
        apply_credential_env(
            username=username,
            api_token=api_token,
            legacy_key=legacy_key,
        )

        try:
            # Validate the candidate credentials before changing persistent state.
            self._run(["kernels", "list", "-m", "-p", "1"], timeout=120)
            self._auth_cache = {
                "fingerprint": self._credential_fingerprint(),
                "ts": time.time(),
            }

            if persist:
                write_credential_env(
                    self.settings.env_file,
                    username=username,
                    api_token=api_token,
                    legacy_key=legacy_key,
                )
        except Exception:
            restore_credential_env(previous_env)
            raise

        resumed = self.resume_waiting_jobs()
        recovered = self.resume_recoverable_outputs()
        return credential_save_message(
            api_token=api_token,
            persist=persist,
            resumed=resumed,
            recovered=recovered,
        )

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def cancel(self, job_id: str) -> str:
        job = self.db.get_job(job_id)
        if not job:
            return "Job introuvable."
        if job.get("status") in {"done", "error", "cancelled", "interrupted"}:
            return f"Job déjà terminé: {job.get('status')}"

        kernel_ref = job.get("kernel_ref") or ""
        meta = job.get("meta") or {}
        dataset_ref = meta.get("dataset_ref") or ""

        with self._lock:
            self._cancelled.add(job_id)
            future = self._futures.get(job_id)
            future_active = bool(future and not future.done())
            cancelled_before_start = bool(future_active and future.cancel())

        # Purely local jobs (not authenticated yet, or queued without an active
        # executor future) can be cancelled synchronously.
        if not kernel_ref and (cancelled_before_start or not future_active):
            self.db.update_job(job_id, status="cancelled", error="")
            with self._lock:
                self._cancelled.discard(job_id)
            return "Job annulé localement. Les entrées sont conservées pour une éventuelle relance."

        self.db.update_job(job_id, status="cancel_requested")

        # Do not make the UI wait for remote deletion. The running worker will
        # also see the cancellation flag and perform final reconciliation.
        if kernel_ref or dataset_ref:
            threading.Thread(
                target=self._cleanup_remote_refs,
                args=(kernel_ref, dataset_ref, True),
                daemon=True,
                name=f"kaggle-cancel-{job_id}",
            ).start()

        return "Annulation demandée. Nettoyage Kaggle en cours en arrière-plan."


    def _check_cancelled(self, job_id: str) -> None:
        if self.is_cancelled(job_id):
            raise JobCancelled("Job annulé par l’utilisateur.")

    def _hash_file(self, path: Path) -> str:
        return input_store_hash_file(path)

    def _input_store_root(self) -> Path:
        return ensure_input_store(self.settings.storage_root)

    def _persist_input(self, job_id: str, source: str | None, stem: str) -> str:
        return persist_input_file(self.settings.storage_root, source)

    def _referenced_input_paths(self) -> set[str]:
        refs: set[str] = set()
        for row in self.db.list_jobs(100000):
            meta = row.get("meta") or {}
            for key in ("source_image", "target_video"):
                value = meta.get(key)
                if value:
                    try:
                        refs.add(str(Path(value).resolve()))
                    except Exception:
                        refs.add(str(Path(value)))
        return refs

    def _gc_input_store(self) -> tuple[int, int]:
        removed = 0
        reclaimed = 0
        for path in unreferenced_store_files(
            self.settings.storage_root,
            referenced_paths=self._referenced_input_paths(),
        ):
            try:
                reclaimed += path.stat().st_size
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed, reclaimed

    def _cleanup_inputs(self, job_id: str, force: bool = False) -> None:
        # Legacy per-job input directories are still removed for old installations.
        if not force:
            if self.settings.keep_job_inputs:
                return
            job = self.db.get_job(job_id)
            meta = (job or {}).get("meta") or {}
            has_source = bool(meta.get("source_image") or meta.get("target_video"))
            if has_source and self.settings.keep_source_inputs_for_retry:
                return
        shutil.rmtree(self.settings.storage_root / "_inputs" / job_id, ignore_errors=True)


    def _validate_submission(
        self,
        task: str,
        prompt: str,
        steps: int,
        cfg: float,
        seed: int,
        aspect: str,
        source_image: str | None = None,
        target_video: str | None = None,
    ) -> None:
        validate_submission(
            task,
            prompt,
            steps,
            cfg,
            seed,
            aspect,
            source_image,
            target_video,
        )

    def submit(
        self,
        task: str,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 25,
        cfg: float = 1.0,
        seed: int = -1,
        aspect: str = "1:1",
        source_image: str | None = None,
        target_video: str | None = None,
    ) -> str:
        self.validate_current_credentials()
        self._validate_submission(
            task, prompt, steps, cfg, seed, aspect, source_image, target_video
        )

        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        persisted_source = self._persist_input(job_id, source_image, "source_image")
        persisted_video = self._persist_input(job_id, target_video, "target_video")
        meta = {
            "negative_prompt": negative_prompt or "",
            "steps": int(steps),
            "cfg": float(cfg),
            "seed": int(seed),
            "aspect": aspect,
            "source_image": persisted_source,
            "target_video": persisted_video,
        }
        self.db.create_job(job_id, task, prompt or "", meta)
        with self._lock:
            self._futures[job_id] = self.executor.submit(self._execute, job_id)
        return job_id

    def _job_config(self, job: dict[str, Any]) -> dict[str, Any]:
        return build_job_config(job)

    def _needs_dataset(self, job: dict[str, Any]) -> bool:
        return needs_dataset(job)

    def _dataset_ref(self, job: dict[str, Any]) -> str:
        return build_dataset_ref(self.settings.kaggle_username, job["id"])

    def _cleanup_remote_refs(
        self,
        kernel_ref: str = "",
        dataset_ref: str = "",
        force: bool = False,
    ) -> None:
        delete_kernel = force or self.settings.delete_remote_kernel
        delete_dataset = force or self.settings.delete_remote_dataset
        if kernel_ref and delete_kernel:
            try:
                self._run(["kernels", "delete", kernel_ref, "-y"], timeout=180)
            except Exception:
                pass
        if dataset_ref and delete_dataset:
            try:
                self._run(["datasets", "delete", dataset_ref, "-y"], timeout=180)
            except Exception:
                pass

    def submit_batch(
        self,
        prompts: str,
        negative_prompt: str = "",
        steps: int = 25,
        cfg: float = 1.0,
        seed: int = -1,
        aspect: str = "1:1",
        max_batch: int = 20,
    ) -> list[str]:
        self.validate_current_credentials()
        self._validate_submission("image", "batch", steps, cfg, seed, aspect)
        lines = [line.strip() for line in (prompts or "").splitlines() if line.strip()]
        if not lines:
            raise ValueError("Aucun prompt dans le lot.")
        if len(lines) > max_batch:
            raise ValueError(f"Maximum {max_batch} prompts par lot.")

        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        meta = {
            "prompts": lines,
            "negative_prompt": negative_prompt or "",
            "steps": int(steps),
            "cfg": float(cfg),
            "seed": int(seed),
            "aspect": aspect,
            "source_image": "",
            "target_video": "",
            "batch_count": len(lines),
        }
        self.db.create_job(
            job_id,
            "image_batch",
            f"Lot de {len(lines)} images",
            meta,
        )
        with self._lock:
            self._futures[job_id] = self.executor.submit(self._execute, job_id)
        return [job_id]


    def _dataset_status(self, dataset_ref: str) -> tuple[str, str]:
        output = self._run(["datasets", "status", dataset_ref], timeout=120)
        return parse_dataset_status(output), output

    def _prepare_dataset(self, job: dict[str, Any], folder: Path, dataset_ref: str | None = None) -> str:
        dataset_ref = dataset_ref or self._dataset_ref(job)
        write_dataset_bundle(
            folder,
            job_id=job["id"],
            dataset_ref=dataset_ref,
            config=self._job_config(job),
        )
        try:
            self._run(
                ["datasets", "create", "-p", str(folder), "-q", "-t", "-r", "skip"],
                timeout=1800,
                retries=1,
            )
        except Exception as create_exc:
            # A timeout can happen after Kaggle accepted the create request.
            # If the expected dataset is visible, continue instead of creating it twice.
            try:
                self._run(["datasets", "status", dataset_ref], timeout=120, retries=2)
            except Exception:
                raise create_exc

        # Dataset creation is asynchronous; wait until it is visible/ready.
        deadline = time.time() + 900
        while time.time() < deadline:
            try:
                state, out = self._dataset_status(dataset_ref)
                if state == "error":
                    raise RuntimeError(out)
                if state == "ready":
                    return dataset_ref
            except RuntimeError as exc:
                if "404" not in str(exc).lower() and "not found" not in str(exc).lower():
                    raise
            time.sleep(5)
        # Some CLI versions do not expose a stable readiness word; returning here
        # lets the kernel push be the final source of truth.
        return dataset_ref

    def _prepare_kernel(self, job: dict[str, Any], folder: Path, dataset_ref: str) -> str:
        kernel_ref, slug = build_kernel_ref(
            self.settings.kaggle_username,
            job["id"],
        )
        write_kernel_bundle(
            folder,
            kernel_ref=kernel_ref,
            title=slug,
            worker_source=self.settings.worker_path.read_text(encoding="utf-8"),
            job_config=self._job_config(job),
            dataset_ref=dataset_ref,
            accelerator=self.settings.accelerator,
        )
        return kernel_ref

    def _download_validated_outputs(
        self,
        job: dict[str, Any],
        kernel_ref: str,
        download: Path,
        attempts: int = 3,
    ) -> tuple[list[tuple[Path, str]], dict[str, Any]]:
        last_exc: Exception | None = None
        total = max(1, int(attempts))
        for attempt in range(1, total + 1):
            self._check_cancelled(job["id"])
            if download.exists():
                shutil.rmtree(download, ignore_errors=True)
            download.mkdir(parents=True, exist_ok=True)
            try:
                self._run(
                    ["kernels", "output", kernel_ref, "-p", str(download), "-o", "-q"],
                    timeout=900,
                )
                staged_artifacts = scan_outputs(download)
                result_manifest = self._validate_downloaded_outputs(job, staged_artifacts)
                return staged_artifacts, result_manifest
            except JobCancelled:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt >= total:
                    break
                # Kaggle output propagation can lag briefly behind a terminal
                # kernel state. Retry from a clean directory so partial files
                # can never be mistaken for a complete download.
                time.sleep(min(15, 3 * attempt))
        raise RuntimeError(
            f"Outputs Kaggle invalides après {total} tentative(s): {last_exc}"
        ) from last_exc

    def _commit_downloaded_outputs(
        self,
        job: dict[str, Any],
        download: Path,
        result_manifest: dict[str, Any],
    ) -> list[tuple[Path, str]]:
        job_id = job["id"]
        artifacts = import_outputs(job_id, download, self.settings.storage_root)

        current = self.db.get_job(job_id) or job
        current_meta = current.get("meta") or {}
        if result_manifest:
            current_meta = {**current_meta, "result_manifest": result_manifest}

        existing = {a["path"] for a in self.db.artifacts(job_id)}
        for path, kind in artifacts:
            if str(path) not in existing:
                self.db.add_artifact(job_id, str(path), kind)

        if not artifacts and not self.db.artifacts(job_id):
            raise RuntimeError("Aucun output Kaggle récupérable.")

        self.db.update_job(
            job_id,
            status="done",
            error="",
            meta_json={
                **current_meta,
                "recover_outputs_available": False,
                "failed_phase": "",
                "auto_recovery_attempts": 0,
            },
        )
        self._storage_cache["ts"] = 0.0
        self._cleanup_inputs(job_id)
        return artifacts

    def _kernel_status(self, kernel_ref: str) -> tuple[str, str]:
        output = self._run(["kernels", "status", kernel_ref], timeout=120)
        return parse_kernel_status(output), output

    def _execute(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if not job:
            return
        kernel_ref = ""
        dataset_ref = ""
        try:
            self._check_cancelled(job_id)
            self.validate_current_credentials()
            self.db.update_job(job_id, status="preparing")
            with tempfile.TemporaryDirectory(prefix=f"qwen-kaggle-{job_id}-") as td:
                work = Path(td)
                dataset_dir = work / "dataset"
                kernel_dir = work / "kernel"
                dataset_dir.mkdir()
                kernel_dir.mkdir()

                self._check_cancelled(job_id)
                if self._needs_dataset(job):
                    dataset_ref = self._dataset_ref(job)
                    self.db.update_job(
                        job_id,
                        status="uploading_inputs",
                        meta_json={**job.get("meta", {}), "dataset_ref": dataset_ref},
                    )
                    dataset_ref = self._prepare_dataset(job, dataset_dir, dataset_ref)
                    self._check_cancelled(job_id)
                else:
                    dataset_ref = ""
                    self.db.update_job(
                        job_id,
                        status="preparing",
                        meta_json={**job.get("meta", {}), "dataset_ref": ""},
                    )

                kernel_ref = self._prepare_kernel(job, kernel_dir, dataset_ref)
                self.db.update_job(
                    job_id,
                    status="submitting",
                    kernel_ref=kernel_ref,
                    meta_json={**job.get("meta", {}), "dataset_ref": dataset_ref},
                )
                self._check_cancelled(job_id)
                try:
                    self._run(
                        [
                            "kernels", "push",
                            "-p", str(kernel_dir),
                            "--accelerator", self.settings.accelerator,
                            "-t", str(self.settings.kernel_timeout),
                        ],
                        timeout=600,
                        retries=1,
                    )
                except Exception as push_exc:
                    # Same protection as dataset creation: after a network timeout,
                    # accept the operation if the exact kernel ref is already visible.
                    try:
                        self._kernel_status(kernel_ref)
                    except Exception:
                        raise push_exc
                self.db.update_job(job_id, status="queued")

                wait_for_kernel(
                    kernel_ref,
                    kernel_timeout=self.settings.kernel_timeout,
                    poll_seconds=self.settings.poll_seconds,
                    check_cancelled=lambda: self._check_cancelled(job_id),
                    kernel_status=self._kernel_status,
                    on_status=lambda state, raw: self.db.update_job(
                        job_id,
                        status=state,
                        meta_json={
                            **job.get("meta", {}),
                            "dataset_ref": dataset_ref,
                            "kaggle_status": raw[-1500:],
                        },
                    ),
                    fetch_logs=lambda ref: self._run(["kernels", "logs", ref], timeout=180),
                    failure_factory=lambda logs: RemoteKernelFailed(
                        "Kaggle a signalé une erreur:\n" + logs[-7000:]
                    ),
                    timeout_message="Le job Kaggle a dépassé le délai maximal.",
                    now=time.time,
                    sleep=time.sleep,
                )

                self._check_cancelled(job_id)
                download = work / "download"
                self.db.update_job(job_id, status="downloading")
                _, result_manifest = self._download_validated_outputs(
                    job, kernel_ref, download
                )
                self._commit_downloaded_outputs(
                    job,
                    download,
                    result_manifest,
                )
        except JobCancelled:
            self.db.update_job(job_id, status="cancelled", error="")
        except Exception as exc:
            if self.is_cancelled(job_id):
                self.db.update_job(job_id, status="cancelled", error="")
            else:
                current = self.db.get_job(job_id) or job
                failed_phase = current.get("status") or ""
                current_meta = current.get("meta") or {}
                auth_required = is_auth_cli_error(str(exc))
                remote_failed = isinstance(exc, RemoteKernelFailed)
                preserve_kernel = preserve_kernel_after_execute_failure(
                    kernel_ref=kernel_ref,
                    failed_phase=failed_phase,
                    remote_failed=remote_failed,
                )
                if auth_required:
                    self.db.update_job(
                        job_id,
                        status="waiting_auth",
                        error=(
                            "Authentification Kaggle requise pour reprendre le kernel distant existant."
                            if kernel_ref
                            else "Authentification Kaggle requise pour reprendre la préparation du job."
                        ),
                        meta_json={
                            **current_meta,
                            "recover_outputs_available": bool(kernel_ref),
                            "failed_phase": "auth_required",
                            "remote_failure_confirmed": False,
                        },
                    )
                else:
                    self.db.update_job(
                        job_id,
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                        meta_json={
                            **current_meta,
                            "recover_outputs_available": preserve_kernel,
                            "failed_phase": failed_phase,
                            "remote_failure_confirmed": remote_failed,
                        },
                    )
        finally:
            current = self.db.get_job(job_id) or {}
            meta = current.get("meta") or {}
            preserve_kernel = bool(meta.get("recover_outputs_available"))
            if preserve_kernel:
                # Inputs dataset is no longer needed once the kernel has completed.
                self._cleanup_remote_refs("", dataset_ref)
            else:
                self._cleanup_remote_refs(kernel_ref, dataset_ref)
            with self._lock:
                self._cancelled.discard(job_id)

    def recover_outputs(self, job_id: str) -> str:
        job = self.db.get_job(job_id)
        if not job:
            raise ValueError("Job introuvable.")
        kernel_ref = job.get("kernel_ref") or ""
        if not kernel_ref:
            raise ValueError("Ce job n'a pas de kernel Kaggle à récupérer.")
        if job.get("status") not in {"error", "interrupted"}:
            raise ValueError("La récupération manuelle est réservée aux jobs en erreur/interrompus.")
        meta = job.get("meta") or {}
        if not has_recoverable_outputs(job):
            raise ValueError(
                "Ce job ne possède pas d'outputs Kaggle conservés. Utilise Relancer pour recalculer."
            )
        with self._lock:
            future = self._futures.get(job_id)
            if future is not None and not future.done():
                raise RuntimeError("Une récupération est déjà en cours pour ce job.")
            self.db.update_job(job_id, status="recovering", error="")
            self._futures[job_id] = self.executor.submit(self._recover_remote_job, job_id)
        return "Récupération des outputs Kaggle relancée sans recalcul GPU."

    def reconcile_completed_jobs(self) -> tuple[int, int]:
        checked = 0
        recoverable = 0
        for row in self.db.list_jobs(100000):
            if row.get("status") != "done":
                continue
            checked += 1
            if has_valid_local_media(self.db.artifacts(row["id"])):
                continue

            kernel_ref = row.get("kernel_ref") or ""
            state = "unknown"
            if kernel_ref:
                try:
                    state, _ = self._kernel_status(kernel_ref)
                except Exception:
                    state = "unknown"

            outcome, can_recover, error = classify_missing_local_outputs(
                kernel_ref=kernel_ref,
                kernel_state=state,
            )
            if outcome == "recoverable":
                meta = row.get("meta") or {}
                self.db.update_job(
                    row["id"],
                    status="error",
                    error=error,
                    meta_json={
                        **meta,
                        "recover_outputs_available": True,
                        "failed_phase": "local_artifacts_missing",
                    },
                )
                recoverable += 1
            else:
                self.db.update_job(
                    row["id"],
                    status="error",
                    error=error,
                )
        return checked, recoverable

    def cleanup_expired_recovery_kernels(self) -> int:
        if not self.credentials_ready():
            return 0
        cutoff = time.time() - max(1, int(self.settings.recovery_retention_days)) * 86400
        removed = 0
        for row in self.db.list_jobs(100000):
            meta = row.get("meta") or {}
            if not recovery_marked(row):
                continue
            if float(row.get("updated_at") or 0) >= cutoff:
                continue
            kernel_ref = row.get("kernel_ref") or ""
            dataset_ref = meta.get("dataset_ref") or ""
            if kernel_ref or dataset_ref:
                try:
                    self._cleanup_remote_refs(kernel_ref, dataset_ref)
                except Exception:
                    continue
            self.db.update_job(
                row["id"],
                status="error",
                error=(
                    f"Délai de récupération dépassé ({self.settings.recovery_retention_days} jours); "
                    "utilise Relancer pour recalculer."
                ),
                meta_json={
                    **meta,
                    "recover_outputs_available": False,
                    "failed_phase": "recovery_expired",
                },
            )
            removed += 1
        return removed

    def reconcile_artifacts(self) -> tuple[int, int]:
        stale_ids: list[int] = []
        checked = 0
        for row in self.db.recent_artifacts(None, 100000):
            checked += 1
            path = row.get("path") or ""
            if not path or not Path(path).is_file():
                stale_ids.append(int(row["id"]))
        removed = self.db.delete_artifacts_by_ids(stale_ids)
        return checked, removed

    def cleanup_storage(self, export_max_age_days: int = 14) -> str:
        now = time.time()
        removed_exports = 0
        removed_inputs = 0
        reclaimed = 0

        export_root = self.settings.storage_root / "_exports"
        for path in expired_exports(
            export_root,
            now=now,
            max_age_days=export_max_age_days,
        ):
            try:
                reclaimed += path.stat().st_size
                path.unlink()
                removed_exports += 1
            except OSError:
                pass

        known_ids = {str(job.get("id")) for job in self.db.list_jobs(100000)}
        input_root = self.settings.storage_root / "_inputs"
        for directory in orphan_input_dirs(input_root, known_ids=known_ids):
            try:
                for path in directory.rglob("*"):
                    if path.is_file():
                        reclaimed += path.stat().st_size
            except OSError:
                pass
            shutil.rmtree(directory, ignore_errors=True)
            removed_inputs += 1

        gc_files, gc_bytes = self._gc_input_store()
        reclaimed += gc_bytes
        _, stale_artifacts = self.reconcile_artifacts()
        _, recoverable_jobs = self.reconcile_completed_jobs()
        expired_recovery_kernels = self.cleanup_expired_recovery_kernels()

        try:
            self.db.vacuum()
        except Exception:
            pass

        self._storage_cache["ts"] = 0.0
        return cleanup_report(
            removed_exports=removed_exports,
            removed_inputs=removed_inputs,
            gc_files=gc_files,
            stale_artifacts=stale_artifacts,
            recoverable_jobs=recoverable_jobs,
            expired_recovery_kernels=expired_recovery_kernels,
            reclaimed=reclaimed,
        )

    def dashboard_summary(self) -> str:
        jobs = self.db.list_jobs(1000)

        now = time.time()
        if now - float(self._storage_cache.get("ts") or 0) > 60:
            total_bytes = 0
            try:
                if self.settings.storage_root.exists():
                    for p in self.settings.storage_root.rglob("*"):
                        if p.is_file():
                            try:
                                total_bytes += p.stat().st_size
                            except OSError:
                                pass
            except Exception:
                pass
            self._storage_cache = {"ts": now, "bytes": total_bytes}
        else:
            total_bytes = int(self._storage_cache.get("bytes") or 0)

        if self.credentials_recently_validated():
            auth = "Kaggle prêt"
        elif self.credentials_ready():
            auth = "Kaggle configuré — à valider"
        else:
            auth = "Kaggle à configurer"

        return dashboard_summary_text(
            jobs,
            total_bytes=total_bytes,
            auth_label=auth,
        )

    def health_check(self) -> str:
        lines = []
        try:
            exe = self.ensure_cli()
            raw = version("kaggle")
            lines.append(f"✅ Kaggle CLI {raw} — {exe}")
        except Exception as exc:
            lines.append(f"❌ Kaggle CLI: {type(exc).__name__}: {exc}")

        try:
            probe = self.settings.storage_root / ".write_test"
            probe.parent.mkdir(parents=True, exist_ok=True)
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            lines.append(f"✅ Stockage accessible — {self.settings.storage_root}")
        except Exception as exc:
            lines.append(f"❌ Stockage: {type(exc).__name__}: {exc}")

        try:
            integrity = self.db.integrity_status()
            if integrity.lower().startswith("ok"):
                lines.append(f"✅ SQLite intègre — {self.settings.db_path}")
                if self.db.recovered_corrupt_path:
                    lines.append(f"ℹ️ Ancienne base corrompue sauvegardée — {self.db.recovered_corrupt_path}")
            else:
                lines.append(f"❌ SQLite integrity_check: {integrity}")
        except Exception as exc:
            lines.append(f"❌ SQLite: {type(exc).__name__}: {exc}")

        if not self.credentials_ready():
            lines.append("⚠️ Kaggle non authentifié.")
        else:
            try:
                self.validate_current_credentials(max_age_seconds=0)
                lines.append("✅ Authentification Kaggle valide.")
            except Exception as exc:
                lines.append(f"❌ Authentification Kaggle: {type(exc).__name__}: {exc}")

        lines.extend(
            recovery_diagnostic_lines(
                self.db.list_jobs(1000),
                recovery_marked=recovery_marked,
                remote_kernel_is_preserved=remote_kernel_is_preserved,
            )
        )

        return "\n".join(lines)

    def remote_logs(self, job_id: str) -> str:
        job = self.db.get_job(job_id)
        if not job:
            return "Job introuvable."
        kernel_ref = job.get("kernel_ref") or ""
        if not kernel_ref:
            return "Ce job n'a pas encore de kernel Kaggle."
        try:
            return self._run(["kernels", "logs", kernel_ref], timeout=180)[-12000:]
        except Exception as exc:
            return f"Logs indisponibles: {type(exc).__name__}: {exc}"

    def queue_position(self, job_id: str) -> int | None:
        active = {
            "preparing", "uploading_inputs", "submitting",
            "queued", "waiting_auth", "running", "recovering", "downloading",
        }
        rows = [
            j for j in self.db.list_jobs(500)
            if j.get("status") in active
        ]
        rows.sort(key=lambda j: float(j.get("created_at") or 0))
        for index, row in enumerate(rows, 1):
            if row.get("id") == job_id:
                return index
        return None

    def job_storage_bytes(self, job_id: str) -> int:
        total = 0
        root = self.settings.storage_root / job_id
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
        return total

    def retry(self, job_id: str) -> str:
        old = self.db.get_job(job_id)
        if not old:
            raise ValueError("Job introuvable.")
        active_states = {
            "preparing", "uploading_inputs", "submitting", "queued",
            "waiting_auth", "running", "recovering", "downloading",
            "cancel_requested",
        }
        if old.get("status") in active_states:
            if old.get("status") == "waiting_auth":
                if old.get("kernel_ref"):
                    raise RuntimeError(
                        "Ce job attend l'authentification pour reprendre son kernel Kaggle existant. "
                        "Configure les identifiants au lieu de le relancer."
                    )
                raise RuntimeError(
                    "Ce job attend l'authentification pour reprendre la file locale. "
                    "Configure les identifiants au lieu de créer un doublon."
                )
            raise RuntimeError("Ce job est encore actif.")

        meta = old.get("meta") or {}
        if old.get("task") == "image_batch":
            prompts = meta.get("prompts") or []
            return self.submit_batch(
                "\n".join(str(x) for x in prompts),
                meta.get("negative_prompt", ""),
                meta.get("steps", 25),
                meta.get("cfg", 1.0),
                meta.get("seed", -1),
                meta.get("aspect", "1:1"),
            )[0]
        return self.submit(
            old.get("task") or "image",
            old.get("prompt") or "",
            meta.get("negative_prompt", ""),
            meta.get("steps", 25),
            meta.get("cfg", 1.0),
            meta.get("seed", -1),
            meta.get("aspect", "1:1"),
            source_image=meta.get("source_image") or None,
            target_video=meta.get("target_video") or None,
        )

    def export_job_archive(self, job_id: str) -> str:
        job = self.db.get_job(job_id)
        if not job:
            raise ValueError("Job introuvable.")
        artifacts = self.db.artifacts(job_id)
        export_root = self.settings.storage_root / "_exports"
        export_root.mkdir(parents=True, exist_ok=True)
        work = export_root / f".{job_id}-export"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        try:
            media_dir = work / "media"
            media_dir.mkdir()
            exported_artifacts = []
            for artifact in artifacts:
                src = Path(artifact.get("path") or "")
                if not src.exists() or not src.is_file():
                    continue
                target = media_dir / src.name
                shutil.copy2(src, target)
                exported_artifacts.append({
                    "name": target.name,
                    "kind": artifact.get("kind") or "file",
                    "size": int(target.stat().st_size),
                    "sha256": self._hash_file(target),
                })
            manifest = {
                "job": job,
                "artifacts": artifacts,
                "exported_artifacts": exported_artifacts,
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (work / "job.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            archive_base = export_root / job_id
            archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=work)
            return archive_path
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def delete_local_job(self, job_id: str) -> str:
        job = self.db.get_job(job_id)
        if not job:
            return "Job introuvable."
        if job.get("status") in {"queued", "running", "submitting", "recovering", "downloading", "cancel_requested"}:
            raise RuntimeError("Annule d'abord le job actif.")
        kernel_ref = job.get("kernel_ref") or ""
        meta = job.get("meta") or {}
        dataset_ref = meta.get("dataset_ref") or ""
        if kernel_ref or dataset_ref:
            try:
                self._cleanup_remote_refs(kernel_ref, dataset_ref)
            except Exception:
                # Local deletion must remain possible if Kaggle is offline or
                # credentials have expired; remote cleanup is best-effort.
                pass
        shutil.rmtree(self.settings.storage_root / job_id, ignore_errors=True)
        self._cleanup_inputs(job_id, force=True)
        try:
            os.remove(self.settings.storage_root / "_exports" / f"{job_id}.zip")
        except OSError:
            pass
        self.db.delete_job(job_id)
        self._gc_input_store()
        self._storage_cache["ts"] = 0.0
        return f"Job {job_id} supprimé du stockage local."

    def job(self, job_id: str):
        return self.db.get_job(job_id)

    def artifacts(self, job_id: str):
        return self.db.artifacts(job_id)

    def recent_artifacts(self, kind: str | None = None, limit: int = 100):
        return self.db.recent_artifacts(kind, limit)

    def jobs(self, limit: int = 100):
        return self.db.list_jobs(limit)

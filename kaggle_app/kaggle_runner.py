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
from .storage import import_outputs

TERMINAL_OK = ("complete", "completed")
TERMINAL_BAD = ("error", "failed", "cancelled", "canceled")

def _is_transient_cli_error(text: str) -> bool:
    low = (text or "").lower()
    transient = (
        "timed out", "timeout", "temporarily unavailable", "connection reset",
        "connection aborted", "connection refused", "remote disconnected",
        "server error", "bad gateway", "gateway timeout", "service unavailable",
        "too many requests", "rate limit", "429", "500", "502", "503", "504",
    )
    permanent = (
        "401", "403", "unauthorized", "forbidden", "invalid token",
        "authentication", "usage:", "unrecognized arguments", "invalid choice",
    )
    if any(token in low for token in permanent):
        return False
    return any(token in low for token in transient)

class JobCancelled(RuntimeError):
    pass

def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    value = re.sub(r"-+", "-", value)
    return value[:48] or "qwen-job"

class KaggleController:
    def __init__(self, settings: Settings = SETTINGS):
        self.settings = settings
        self.db = JobDB(settings.db_path)
        # Kaggle GPU sessions are intentionally serialized.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kaggle-job")
        self._futures: dict[str, Any] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()
        self._recover_persisted_jobs()

    def _recover_persisted_jobs(self) -> None:
        recoverable = {"submitting", "queued", "running", "recovering", "downloading"}
        interrupted = {"preparing", "uploading_inputs"}
        for row in self.db.list_jobs(200):
            job_id = row.get("id")
            status = row.get("status")
            kernel_ref = row.get("kernel_ref") or ""
            if status in interrupted:
                meta = row.get("meta") or {}
                dataset_ref = meta.get("dataset_ref") or ""
                self.db.update_job(
                    job_id,
                    status="interrupted",
                    error="Le contrôleur s'est arrêté avant la création complète du kernel Kaggle. Relance ce job.",
                )
                if dataset_ref:
                    self.executor.submit(self._cleanup_remote_refs, "", dataset_ref)
            elif status in recoverable and kernel_ref:
                with self._lock:
                    self._futures[job_id] = self.executor.submit(self._recover_remote_job, job_id)

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
            self.db.update_job(job_id, status="recovering")
            started = time.time()
            while True:
                self._check_cancelled(job_id)
                if time.time() - started > self.settings.kernel_timeout + 1200:
                    raise TimeoutError("Délai maximal dépassé pendant la récupération Kaggle.")
                state, raw = self._kernel_status(kernel_ref)
                self.db.update_job(
                    job_id,
                    status=("recovering" if state in {"queued", "running"} else state),
                    meta_json={**meta, "dataset_ref": dataset_ref, "kaggle_status": raw[-1500:]},
                )
                if state == "complete":
                    break
                if state == "error":
                    try:
                        logs = self._run(["kernels", "logs", kernel_ref], timeout=180)
                    except Exception:
                        logs = raw
                    raise RuntimeError("Le job Kaggle récupéré a échoué:\n" + logs[-7000:])
                time.sleep(max(5, self.settings.poll_seconds))

            with tempfile.TemporaryDirectory(prefix=f"qwen-kaggle-recover-{job_id}-") as td:
                download = Path(td) / "download"
                download.mkdir(parents=True, exist_ok=True)
                self.db.update_job(job_id, status="downloading")
                self._run(
                    ["kernels", "output", kernel_ref, "-p", str(download), "-o", "-q"],
                    timeout=900,
                )
                artifacts = import_outputs(job_id, download, self.settings.storage_root)
                existing = {a["path"] for a in self.db.artifacts(job_id)}
                for path, kind in artifacts:
                    if str(path) not in existing:
                        self.db.add_artifact(job_id, str(path), kind)
                if not artifacts and not self.db.artifacts(job_id):
                    raise RuntimeError("Aucun output Kaggle récupérable.")
                self.db.update_job(job_id, status="done", error="")
                self._cleanup_inputs(job_id)
        except JobCancelled:
            self.db.update_job(job_id, status="cancelled", error="")
        except Exception as exc:
            if self.is_cancelled(job_id):
                self.db.update_job(job_id, status="cancelled", error="")
            else:
                self.db.update_job(
                    job_id,
                    status="error",
                    error=f"Recovery {type(exc).__name__}: {exc}",
                )
        finally:
            self._cleanup_remote_refs(kernel_ref, dataset_ref)
            with self._lock:
                self._cancelled.discard(job_id)

    def credentials_ready(self) -> bool:
        return bool(
            self.settings.kaggle_username
            and (self.settings.kaggle_api_token or self.settings.kaggle_key)
        )

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

    def _run(self, args: list[str], timeout: int | None = None) -> str:
        exe = self.ensure_cli()
        attempts = max(1, int(self.settings.cli_retries))
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
                if attempt >= attempts or not _is_transient_cli_error(output):
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
        if not username:
            raise ValueError("KAGGLE_USERNAME est requis pour créer les kernels/datasets.")
        if not api_token and not legacy_key:
            raise ValueError("Ajoute KAGGLE_API_TOKEN (recommandé) ou l'ancien KAGGLE_KEY.")

        os.environ["KAGGLE_USERNAME"] = username
        if api_token:
            os.environ["KAGGLE_API_TOKEN"] = api_token
            os.environ.pop("KAGGLE_KEY", None)
        else:
            os.environ["KAGGLE_KEY"] = legacy_key
            os.environ.pop("KAGGLE_API_TOKEN", None)

        if persist:
            env_path = self.settings.env_file
            lines = [f"KAGGLE_USERNAME={username}"]
            if api_token:
                lines.append(f"KAGGLE_API_TOKEN={api_token}")
            else:
                lines.append(f"KAGGLE_KEY={legacy_key}")
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            try:
                os.chmod(env_path, 0o600)
            except Exception:
                pass

        self._run(["kernels", "list", "-m", "-p", "1"], timeout=120)
        mode = "API token" if api_token else "legacy key"
        return (
            f"Identifiants Kaggle validés ({mode})"
            + (" et sauvegardés dans le stockage privé configuré." if persist else ".")
        )

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def cancel(self, job_id: str) -> str:
        job = self.db.get_job(job_id)
        if not job:
            return "Job introuvable."
        if job.get("status") in {"done", "error", "cancelled"}:
            return f"Job déjà terminé: {job.get('status')}"

        with self._lock:
            self._cancelled.add(job_id)
            future = self._futures.get(job_id)
            cancelled_before_start = bool(future and future.cancel())

        self.db.update_job(job_id, status="cancel_requested")
        if cancelled_before_start:
            self.db.update_job(job_id, status="cancelled")
            return "Job annulé avant son démarrage. Les entrées sont conservées pour une éventuelle relance."

        kernel_ref = job.get("kernel_ref") or ""
        meta = job.get("meta") or {}
        dataset_ref = meta.get("dataset_ref") or ""
        if kernel_ref:
            try:
                self._run(["kernels", "delete", kernel_ref, "-y"], timeout=180)
            except Exception:
                pass
        if dataset_ref:
            try:
                self._run(["datasets", "delete", dataset_ref, "-y"], timeout=180)
            except Exception:
                pass
        return "Annulation demandée. Le worker local finalise le nettoyage."

    def _check_cancelled(self, job_id: str) -> None:
        if self.is_cancelled(job_id):
            raise JobCancelled("Job annulé par l’utilisateur.")

    def _persist_input(self, job_id: str, source: str | None, stem: str) -> str:
        if not source:
            return ""
        src = Path(source)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Fichier uploadé introuvable: {src}")
        dest_dir = self.settings.storage_root / "_inputs" / job_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        suffix = src.suffix.lower()
        dest = dest_dir / f"{stem}{suffix}"
        shutil.copy2(src, dest)
        return str(dest)

    def _cleanup_inputs(self, job_id: str) -> None:
        if self.settings.keep_job_inputs:
            return
        shutil.rmtree(self.settings.storage_root / "_inputs" / job_id, ignore_errors=True)

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
        if not self.credentials_ready():
            raise RuntimeError("Configure d'abord KAGGLE_USERNAME et KAGGLE_KEY.")
        if task in {"image", "image_edit"} and not (prompt or "").strip():
            raise ValueError("Le prompt est vide.")
        if task == "image_edit" and not source_image:
            raise ValueError("image_edit requiert une image source.")
        if task == "video_faceswap" and (not source_image or not target_video):
            raise ValueError("video_faceswap requiert un visage source et une vidéo cible.")

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

    def _dataset_ref(self, job: dict[str, Any]) -> str:
        username = self.settings.kaggle_username
        dataset_slug = slugify(f"qwen-input-{job['id']}")
        return f"{username}/{dataset_slug}"

    def _cleanup_remote_refs(self, kernel_ref: str = "", dataset_ref: str = "") -> None:
        if not self.settings.delete_remote_kernel:
            return
        if kernel_ref:
            try:
                self._run(["kernels", "delete", kernel_ref, "-y"], timeout=180)
            except Exception:
                pass
        if dataset_ref:
            try:
                self._run(["datasets", "delete", dataset_ref, "-y"], timeout=180)
            except Exception:
                pass

    def _prepare_dataset(self, job: dict[str, Any], folder: Path, dataset_ref: str | None = None) -> str:
        dataset_ref = dataset_ref or self._dataset_ref(job)
        config = {
            "job_id": job["id"],
            "task": job["task"],
            "prompt": job.get("prompt", ""),
            **job.get("meta", {}),
        }
        for key in ("source_image", "target_video"):
            src = config.get(key)
            if src:
                p = Path(src)
                if not p.exists():
                    raise FileNotFoundError(f"Fichier d'entrée introuvable: {p}")
                target = folder / ("source_image" + p.suffix if key == "source_image" else "target_video" + p.suffix)
                shutil.copy2(p, target)
                config[key] = target.name

        (folder / "job_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metadata = {
            "title": f"Qwen input {job['id']}"[:50],
            "id": dataset_ref,
            "licenses": [{"name": "other"}],
        }
        (folder / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._run(["datasets", "create", "-p", str(folder), "-q", "-t", "-r", "skip"], timeout=1800)

        # Dataset creation is asynchronous; wait until it is visible/ready.
        deadline = time.time() + 900
        while time.time() < deadline:
            try:
                out = self._run(["datasets", "status", dataset_ref], timeout=120)
                low = out.lower()
                if "error" in low or "failed" in low:
                    raise RuntimeError(out)
                if "ready" in low or "complete" in low or "completed" in low:
                    return dataset_ref
            except RuntimeError as exc:
                if "404" not in str(exc).lower() and "not found" not in str(exc).lower():
                    raise
            time.sleep(5)
        # Some CLI versions do not expose a stable readiness word; returning here
        # lets the kernel push be the final source of truth.
        return dataset_ref

    def _prepare_kernel(self, job: dict[str, Any], folder: Path, dataset_ref: str) -> str:
        username = self.settings.kaggle_username
        slug = slugify(f"qwen-studio-{job['id']}")
        kernel_ref = f"{username}/{slug}"

        worker_source = self.settings.worker_path.read_text(encoding="utf-8")
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Qwen Kaggle Worker\n",
                        "Notebook généré automatiquement par Qwen Kaggle Studio.\n",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": worker_source.splitlines(keepends=True),
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (folder / "job.ipynb").write_text(
            json.dumps(notebook, ensure_ascii=False),
            encoding="utf-8",
        )

        metadata = {
            "id": kernel_ref,
            "title": slug,
            "code_file": "job.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": self.settings.accelerator,
            "dataset_sources": [dataset_ref],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return kernel_ref

    def _kernel_status(self, kernel_ref: str) -> tuple[str, str]:
        output = self._run(["kernels", "status", kernel_ref], timeout=120)
        low = output.lower()
        if any(x in low for x in TERMINAL_BAD):
            return "error", output
        if any(x in low for x in TERMINAL_OK):
            return "complete", output
        if "queued" in low or "pending" in low:
            return "queued", output
        return "running", output

    def _execute(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if not job:
            return
        kernel_ref = ""
        dataset_ref = ""
        try:
            self._check_cancelled(job_id)
            self.db.update_job(job_id, status="preparing")
            with tempfile.TemporaryDirectory(prefix=f"qwen-kaggle-{job_id}-") as td:
                work = Path(td)
                dataset_dir = work / "dataset"
                kernel_dir = work / "kernel"
                dataset_dir.mkdir()
                kernel_dir.mkdir()

                self._check_cancelled(job_id)
                dataset_ref = self._dataset_ref(job)
                self.db.update_job(
                    job_id,
                    status="uploading_inputs",
                    meta_json={**job.get("meta", {}), "dataset_ref": dataset_ref},
                )
                dataset_ref = self._prepare_dataset(job, dataset_dir, dataset_ref)
                self._check_cancelled(job_id)

                kernel_ref = self._prepare_kernel(job, kernel_dir, dataset_ref)
                self.db.update_job(
                    job_id,
                    status="submitting",
                    kernel_ref=kernel_ref,
                    meta_json={**job.get("meta", {}), "dataset_ref": dataset_ref},
                )
                self._check_cancelled(job_id)
                self._run(
                    [
                        "kernels", "push",
                        "-p", str(kernel_dir),
                        "--accelerator", self.settings.accelerator,
                        "-t", str(self.settings.kernel_timeout),
                    ],
                    timeout=600,
                )
                self.db.update_job(job_id, status="queued")

                started = time.time()
                while True:
                    self._check_cancelled(job_id)
                    if time.time() - started > self.settings.kernel_timeout + 1200:
                        raise TimeoutError("Le job Kaggle a dépassé le délai maximal.")
                    state, raw = self._kernel_status(kernel_ref)
                    self.db.update_job(
                        job_id,
                        status=state,
                        meta_json={
                            **job.get("meta", {}),
                            "dataset_ref": dataset_ref,
                            "kaggle_status": raw[-1500:],
                        },
                    )
                    if state == "complete":
                        break
                    if state == "error":
                        try:
                            logs = self._run(["kernels", "logs", kernel_ref], timeout=180)
                        except Exception:
                            logs = raw
                        raise RuntimeError("Kaggle a signalé une erreur:\n" + logs[-7000:])
                    time.sleep(max(5, self.settings.poll_seconds))

                self._check_cancelled(job_id)
                download = work / "download"
                download.mkdir(parents=True, exist_ok=True)
                self.db.update_job(job_id, status="downloading")
                self._run(
                    ["kernels", "output", kernel_ref, "-p", str(download), "-o", "-q"],
                    timeout=900,
                )

                artifacts = import_outputs(job_id, download, self.settings.storage_root)
                for path, kind in artifacts:
                    self.db.add_artifact(job_id, str(path), kind)
                if not artifacts:
                    raise RuntimeError("Le kernel Kaggle s'est terminé sans produire de fichier.")
                self.db.update_job(job_id, status="done")
                self._cleanup_inputs(job_id)
        except JobCancelled:
            self.db.update_job(job_id, status="cancelled", error="")
        except Exception as exc:
            if self.is_cancelled(job_id):
                self.db.update_job(job_id, status="cancelled", error="")
            else:
                self.db.update_job(job_id, status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            if kernel_ref and self.settings.delete_remote_kernel:
                try:
                    self._run(["kernels", "delete", kernel_ref, "-y"], timeout=180)
                except Exception:
                    pass
            if dataset_ref and self.settings.delete_remote_kernel:
                try:
                    self._run(["datasets", "delete", dataset_ref, "-y"], timeout=180)
                except Exception:
                    pass
            with self._lock:
                self._cancelled.discard(job_id)

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
            with self.db._conn() as con:
                con.execute("SELECT 1").fetchone()
            lines.append(f"✅ SQLite accessible — {self.settings.db_path}")
        except Exception as exc:
            lines.append(f"❌ SQLite: {type(exc).__name__}: {exc}")

        if not self.credentials_ready():
            lines.append("⚠️ Kaggle non authentifié.")
        else:
            try:
                self._run(["kernels", "list", "-m", "-p", "1"], timeout=120)
                lines.append("✅ Authentification Kaggle valide.")
            except Exception as exc:
                lines.append(f"❌ Authentification Kaggle: {type(exc).__name__}: {exc}")

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
            "queued", "running", "recovering", "downloading",
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
        if old.get("status") in {"queued", "running", "submitting", "recovering", "downloading"}:
            raise RuntimeError("Ce job est encore actif.")

        meta = old.get("meta") or {}
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

    def delete_local_job(self, job_id: str) -> str:
        job = self.db.get_job(job_id)
        if not job:
            return "Job introuvable."
        if job.get("status") in {"queued", "running", "submitting", "recovering", "downloading", "cancel_requested"}:
            raise RuntimeError("Annule d'abord le job actif.")
        shutil.rmtree(self.settings.storage_root / job_id, ignore_errors=True)
        shutil.rmtree(self.settings.storage_root / "_inputs" / job_id, ignore_errors=True)
        self.db.delete_job(job_id)
        return f"Job {job_id} supprimé du stockage local."

    def job(self, job_id: str):
        return self.db.get_job(job_id)

    def artifacts(self, job_id: str):
        return self.db.artifacts(job_id)

    def jobs(self, limit: int = 100):
        return self.db.list_jobs(limit)

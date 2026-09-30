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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import SETTINGS, Settings
from .db import JobDB
from .storage import import_outputs

TERMINAL_OK = ("complete", "completed")
TERMINAL_BAD = ("error", "failed", "cancelled", "canceled")

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
        self._lock = threading.RLock()

    def credentials_ready(self) -> bool:
        return bool(self.settings.kaggle_username and self.settings.kaggle_key)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.settings.kaggle_username:
            env["KAGGLE_USERNAME"] = self.settings.kaggle_username
        if self.settings.kaggle_key:
            env["KAGGLE_KEY"] = self.settings.kaggle_key
        return env

    def ensure_cli(self) -> str:
        exe = shutil.which("kaggle")
        if exe:
            return exe
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--quiet", "kaggle"],
            check=True,
            timeout=300,
        )
        exe = shutil.which("kaggle")
        if not exe:
            raise RuntimeError("Le CLI Kaggle n'a pas été trouvé après installation.")
        return exe

    def _run(self, args: list[str], timeout: int | None = None) -> str:
        exe = self.ensure_cli()
        p = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            env=self._env(),
            timeout=timeout,
        )
        output = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
        if p.returncode != 0:
            raise RuntimeError(output[-7000:] or f"Kaggle CLI exit={p.returncode}")
        return output

    def save_credentials(self, username: str, key: str, persist: bool = True) -> str:
        username = (username or "").strip()
        key = (key or "").strip()
        if not username or not key:
            raise ValueError("KAGGLE_USERNAME et KAGGLE_KEY sont requis.")
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        if persist:
            env_path = self.settings.root / ".env.local"
            env_path.write_text(
                f"KAGGLE_USERNAME={username}\nKAGGLE_KEY={key}\n",
                encoding="utf-8",
            )
            try:
                os.chmod(env_path, 0o600)
            except Exception:
                pass
        # Force a real authenticated call now, not only at job submission.
        self._run(["kernels", "list", "-m", "--page-size", "1"], timeout=120)
        return "Identifiants Kaggle validés" + (" et sauvegardés dans .env.local." if persist else ".")

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
        if task == "image" and not (prompt or "").strip():
            raise ValueError("Le prompt est vide.")
        if task == "video_faceswap" and (not source_image or not target_video):
            raise ValueError("video_faceswap requiert un visage source et une vidéo cible.")

        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        meta = {
            "negative_prompt": negative_prompt or "",
            "steps": int(steps),
            "cfg": float(cfg),
            "seed": int(seed),
            "aspect": aspect,
            "source_image": source_image or "",
            "target_video": target_video or "",
        }
        self.db.create_job(job_id, task, prompt or "", meta)
        with self._lock:
            self._futures[job_id] = self.executor.submit(self._execute, job_id)
        return job_id

    def _prepare_dataset(self, job: dict[str, Any], folder: Path) -> str:
        username = self.settings.kaggle_username
        dataset_slug = slugify(f"qwen-input-{job['id']}")
        dataset_ref = f"{username}/{dataset_slug}"
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
            "licenses": [{"name": "CC0-1.0"}],
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
            self.db.update_job(job_id, status="preparing")
            with tempfile.TemporaryDirectory(prefix=f"qwen-kaggle-{job_id}-") as td:
                work = Path(td)
                dataset_dir = work / "dataset"
                kernel_dir = work / "kernel"
                dataset_dir.mkdir()
                kernel_dir.mkdir()

                self.db.update_job(job_id, status="uploading_inputs")
                dataset_ref = self._prepare_dataset(job, dataset_dir)

                kernel_ref = self._prepare_kernel(job, kernel_dir, dataset_ref)
                self.db.update_job(
                    job_id,
                    status="submitting",
                    kernel_ref=kernel_ref,
                    meta_json={**job.get("meta", {}), "dataset_ref": dataset_ref},
                )
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
        except Exception as exc:
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

    def job(self, job_id: str):
        return self.db.get_job(job_id)

    def artifacts(self, job_id: str):
        return self.db.artifacts(job_id)

    def jobs(self, limit: int = 100):
        return self.db.list_jobs(limit)

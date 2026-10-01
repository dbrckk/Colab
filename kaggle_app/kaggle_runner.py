from __future__ import annotations

import json
import hashlib
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
from .storage import import_outputs, scan_outputs

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
        self._storage_cache = {"ts": 0.0, "bytes": 0}
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
            elif status in {"queued", "waiting_auth"} and not kernel_ref:
                if self.credentials_ready():
                    self.db.update_job(job_id, status="queued", error="")
                    with self._lock:
                        self._futures[job_id] = self.executor.submit(self._execute, job_id)
                else:
                    self.db.update_job(
                        job_id,
                        status="waiting_auth",
                        error="En attente des identifiants Kaggle pour reprendre la file locale.",
                    )

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
        if not artifacts:
            raise RuntimeError("Le kernel Kaggle s'est terminé sans produire de fichier.")

        expected_kind = {
            "image": "image",
            "image_edit": "image",
            "image_batch": "image",
            "video_faceswap": "video",
        }.get(job.get("task"))

        data: dict[str, Any] = {}
        manifests = [path for path, kind in artifacts if path.name == "result.json"]
        if manifests:
            try:
                data = json.loads(manifests[-1].read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"result.json illisible: {exc}") from exc
            if data.get("status") != "done":
                raise RuntimeError(
                    "Le worker Kaggle n'a pas confirmé la réussite: "
                    + str(data.get("error") or data.get("status") or "statut inconnu")
                )

        artifact_names = {path.name for path, _ in artifacts}
        declared_files = [str(x) for x in (data.get("files") or [])]
        missing_declared = [name for name in declared_files if Path(name).name not in artifact_names]
        if missing_declared:
            raise RuntimeError(
                "Le manifeste Kaggle référence des fichiers absents: "
                + ", ".join(missing_declared[:10])
            )

        if expected_kind:
            media = [
                path for path, kind in artifacts
                if kind == expected_kind and self._media_integrity_ok(path, kind)
            ]
            if not media:
                raise RuntimeError(
                    f"Le job {job.get('task')} est terminé sans média {expected_kind} décodable."
                )
        return data

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
                staged_artifacts = scan_outputs(download)
                result_manifest = self._validate_downloaded_outputs(job, staged_artifacts)
                artifacts = import_outputs(job_id, download, self.settings.storage_root)
                current = self.db.get_job(job_id) or job
                current_meta = current.get("meta") or {}
                if result_manifest:
                    self.db.update_job(
                        job_id,
                        meta_json={**current_meta, "result_manifest": result_manifest},
                    )
                existing = {a["path"] for a in self.db.artifacts(job_id)}
                for path, kind in artifacts:
                    if str(path) not in existing:
                        self.db.add_artifact(job_id, str(path), kind)
                if not artifacts and not self.db.artifacts(job_id):
                    raise RuntimeError("Aucun output Kaggle récupérable.")
                self.db.update_job(job_id, status="done", error="")
                self._storage_cache["ts"] = 0.0
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

    def resume_waiting_jobs(self) -> int:
        if not self.credentials_ready():
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
                self.db.update_job(job_id, status="queued", error="")
                self._futures[job_id] = self.executor.submit(self._execute, job_id)
                resumed += 1
        return resumed

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
        resumed = self.resume_waiting_jobs()
        mode = "API token" if api_token else "legacy key"
        suffix = f" {resumed} job(s) en attente relancé(s)." if resumed else ""
        return (
            f"Identifiants Kaggle validés ({mode})"
            + (" et sauvegardés dans le stockage privé configuré." if persist else ".")
            + suffix
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
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _input_store_root(self) -> Path:
        root = self.settings.storage_root / "_input_store"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _persist_input(self, job_id: str, source: str | None, stem: str) -> str:
        if not source:
            return ""
        src = Path(source)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Fichier uploadé introuvable: {src}")

        store = self._input_store_root()
        try:
            if src.parent.resolve() == store.resolve():
                return str(src)
        except Exception:
            pass

        digest = self._hash_file(src)
        suffix = src.suffix.lower()

        # Same bytes may arrive under a renamed extension; reuse any existing
        # canonical object with the same SHA-256 rather than storing it twice.
        existing = next(
            (
                candidate for candidate in store.glob(f"{digest}.*")
                if candidate.is_file() and not candidate.name.startswith(".")
            ),
            None,
        )
        if existing is not None:
            return str(existing)

        dest = store / f"{digest}{suffix}"
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            tmp = store / f".{digest}.{uuid.uuid4().hex[:8]}.part"
            try:
                shutil.copy2(src, tmp)
                if self._hash_file(tmp) != digest:
                    raise RuntimeError("Checksum invalide après copie de l'upload.")
                os.replace(tmp, dest)
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
        return str(dest)

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
        store = self.settings.storage_root / "_input_store"
        if not store.exists():
            return 0, 0
        refs = self._referenced_input_paths()
        removed = 0
        reclaimed = 0
        for p in store.iterdir():
            if not p.is_file() or p.name.startswith("."):
                continue
            try:
                resolved = str(p.resolve())
            except Exception:
                resolved = str(p)
            if resolved in refs:
                continue
            try:
                reclaimed += p.stat().st_size
                p.unlink()
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
            raise RuntimeError("Configure d'abord KAGGLE_USERNAME et KAGGLE_API_TOKEN (ou KAGGLE_KEY legacy).")
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

    def _job_config(self, job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": job["id"],
            "task": job["task"],
            "prompt": job.get("prompt", ""),
            **(job.get("meta") or {}),
        }

    def _needs_dataset(self, job: dict[str, Any]) -> bool:
        meta = job.get("meta") or {}
        return bool(meta.get("source_image") or meta.get("target_video"))

    def _dataset_ref(self, job: dict[str, Any]) -> str:
        username = self.settings.kaggle_username
        dataset_slug = slugify(f"qwen-input-{job['id']}")
        return f"{username}/{dataset_slug}"

    def _cleanup_remote_refs(
        self,
        kernel_ref: str = "",
        dataset_ref: str = "",
        force: bool = False,
    ) -> None:
        if not force and not self.settings.delete_remote_kernel:
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
        if not self.credentials_ready():
            raise RuntimeError(
                "Configure d'abord KAGGLE_USERNAME et KAGGLE_API_TOKEN "
                "(ou KAGGLE_KEY legacy)."
            )
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


    def _prepare_dataset(self, job: dict[str, Any], folder: Path, dataset_ref: str | None = None) -> str:
        dataset_ref = dataset_ref or self._dataset_ref(job)
        config = self._job_config(job)
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
        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Qwen Kaggle Worker\n",
                    "Notebook généré automatiquement par Qwen Kaggle Studio.\n",
                ],
            }
        ]
        if not dataset_ref:
            inline_config = json.dumps(self._job_config(job), ensure_ascii=False)
            bootstrap = (
                "import json\n"
                "from pathlib import Path\n"
                f"_config = {inline_config!r}\n"
                "Path('/kaggle/working/job_config.json').write_text(_config, encoding='utf-8')\n"
            )
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": bootstrap.splitlines(keepends=True),
            })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": worker_source.splitlines(keepends=True),
        })
        notebook = {
            "cells": cells,
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
            "dataset_sources": [dataset_ref] if dataset_ref else [],
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

                staged_artifacts = scan_outputs(download)
                result_manifest = self._validate_downloaded_outputs(job, staged_artifacts)
                artifacts = import_outputs(job_id, download, self.settings.storage_root)
                current = self.db.get_job(job_id) or job
                current_meta = current.get("meta") or {}
                if result_manifest:
                    self.db.update_job(
                        job_id,
                        meta_json={**current_meta, "result_manifest": result_manifest},
                    )
                for path, kind in artifacts:
                    self.db.add_artifact(job_id, str(path), kind)
                self.db.update_job(job_id, status="done")
                self._storage_cache["ts"] = 0.0
                self._cleanup_inputs(job_id)
        except JobCancelled:
            self.db.update_job(job_id, status="cancelled", error="")
        except Exception as exc:
            if self.is_cancelled(job_id):
                self.db.update_job(job_id, status="cancelled", error="")
            else:
                self.db.update_job(job_id, status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            self._cleanup_remote_refs(kernel_ref, dataset_ref)
            with self._lock:
                self._cancelled.discard(job_id)

    def cleanup_storage(self, export_max_age_days: int = 14) -> str:
        now = time.time()
        removed_exports = 0
        removed_inputs = 0
        reclaimed = 0

        export_root = self.settings.storage_root / "_exports"
        if export_root.exists():
            cutoff = now - max(1, int(export_max_age_days)) * 86400
            for p in export_root.glob("*.zip"):
                try:
                    if p.stat().st_mtime < cutoff:
                        reclaimed += p.stat().st_size
                        p.unlink()
                        removed_exports += 1
                except OSError:
                    pass

        known_ids = {str(j.get("id")) for j in self.db.list_jobs(100000)}
        input_root = self.settings.storage_root / "_inputs"
        if input_root.exists():
            for d in input_root.iterdir():
                if not d.is_dir() or d.name in known_ids:
                    continue
                try:
                    for p in d.rglob("*"):
                        if p.is_file():
                            reclaimed += p.stat().st_size
                except OSError:
                    pass
                shutil.rmtree(d, ignore_errors=True)
                removed_inputs += 1

        gc_files, gc_bytes = self._gc_input_store()
        reclaimed += gc_bytes

        try:
            self.db.vacuum()
        except Exception:
            pass

        amount = (
            f"{reclaimed / 1024**3:.2f} Go"
            if reclaimed >= 1024**3
            else f"{reclaimed / 1024**2:.1f} Mo"
        )
        self._storage_cache["ts"] = 0.0
        return (
            f"Nettoyage terminé • {removed_exports} export(s) ancien(s) • "
            f"{removed_inputs} dossier(s) d'entrée orphelin(s) • "
            f"{gc_files} source(s) partagée(s) non référencée(s) • {amount} libéré(s)."
        )

    def dashboard_summary(self) -> str:
        jobs = self.db.list_jobs(1000)
        active_states = {
            "preparing", "uploading_inputs", "submitting",
            "queued", "waiting_auth", "running", "recovering", "downloading",
            "cancel_requested",
        }
        active = sum(1 for j in jobs if j.get("status") in active_states)
        done = sum(1 for j in jobs if j.get("status") == "done")
        failed = sum(1 for j in jobs if j.get("status") in {"error", "interrupted"})
        cancelled = sum(1 for j in jobs if j.get("status") == "cancelled")

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

        if total_bytes >= 1024**3:
            storage = f"{total_bytes / 1024**3:.2f} Go"
        else:
            storage = f"{total_bytes / 1024**2:.1f} Mo"

        auth = "Kaggle prêt" if self.credentials_ready() else "Kaggle à configurer"
        return (
            f"**{auth}**  •  "
            f"Actifs **{active}**  •  Terminés **{done}**  •  "
            f"Erreurs **{failed}**  •  Annulés **{cancelled}**  •  "
            f"Stockage **{storage}**"
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
        if old.get("status") in {"queued", "running", "submitting", "recovering", "downloading"}:
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
            manifest = {
                "job": job,
                "artifacts": artifacts,
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (work / "job.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            media_dir = work / "media"
            media_dir.mkdir()
            for artifact in artifacts:
                src = Path(artifact.get("path") or "")
                if src.exists() and src.is_file():
                    shutil.copy2(src, media_dir / src.name)
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

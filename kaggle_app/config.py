from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    storage_root: Path = Path(os.getenv("QWEN_KAGGLE_STORAGE", ROOT / "storage" / "kaggle_media"))
    db_path: Path = Path(os.getenv("QWEN_KAGGLE_DB", ROOT / "storage" / "kaggle_jobs.sqlite3"))
    env_file: Path = Path(os.getenv("QWEN_KAGGLE_ENV_FILE", ROOT / ".env.local"))
    worker_path: Path = ROOT / "kaggle_worker" / "worker.py"
    poll_seconds: int = int(os.getenv("KAGGLE_POLL_SECONDS", "20"))
    cli_retries: int = int(os.getenv("KAGGLE_CLI_RETRIES", "3"))
    kernel_timeout: int = int(os.getenv("KAGGLE_KERNEL_TIMEOUT", "21600"))
    accelerator: str = os.getenv("KAGGLE_ACCELERATOR", "NvidiaTeslaT4")
    delete_remote_kernel: bool = _bool("KAGGLE_DELETE_REMOTE_KERNEL", True)
    recovery_retention_days: int = max(1, int(os.getenv("KAGGLE_RECOVERY_RETENTION_DAYS", "7")))
    keep_job_inputs: bool = _bool("QWEN_KEEP_JOB_INPUTS", False)
    keep_source_inputs_for_retry: bool = _bool("QWEN_KEEP_SOURCE_INPUTS_FOR_RETRY", True)
    share_gradio: bool = _bool("QWEN_KAGGLE_SHARE", True)

    @property
    def kaggle_username(self) -> str:
        return os.getenv("KAGGLE_USERNAME", "").strip()

    @property
    def kaggle_api_token(self) -> str:
        return os.getenv("KAGGLE_API_TOKEN", "").strip()

    @property
    def kaggle_key(self) -> str:
        return os.getenv("KAGGLE_KEY", "").strip()

    def ensure_dirs(self) -> None:
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.env_file.parent.mkdir(parents=True, exist_ok=True)

SETTINGS = Settings()
SETTINGS.ensure_dirs()

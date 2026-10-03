from __future__ import annotations

import re
from typing import Any


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9-]+", "-", (value or "").lower()).strip("-")
    value = re.sub(r"-+", "-", value)
    return value[:48] or "qwen-job"


def build_job_config(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["id"],
        "task": job["task"],
        "prompt": job.get("prompt", ""),
        **(job.get("meta") or {}),
    }


def needs_dataset(job: dict[str, Any]) -> bool:
    meta = job.get("meta") or {}
    return bool(meta.get("source_image") or meta.get("target_video"))


def dataset_ref(username: str, job_id: str) -> str:
    return f"{username}/{slugify(f'qwen-input-{job_id}')}"


def kernel_ref(username: str, job_id: str) -> tuple[str, str]:
    slug = slugify(f"qwen-studio-{job_id}")
    return f"{username}/{slug}", slug

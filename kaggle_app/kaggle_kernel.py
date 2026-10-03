from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_kernel_bundle(
    folder: Path,
    *,
    kernel_ref: str,
    title: str,
    worker_source: str,
    job_config: dict[str, Any],
    dataset_ref: str,
    accelerator: str,
) -> None:
    cells: list[dict[str, Any]] = [
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
        inline_config = json.dumps(job_config, ensure_ascii=False)
        bootstrap = (
            "import json\n"
            "from pathlib import Path\n"
            f"_config = {inline_config!r}\n"
            "Path('/kaggle/working/job_config.json').write_text(_config, encoding='utf-8')\n"
        )
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": bootstrap.splitlines(keepends=True),
            }
        )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": worker_source.splitlines(keepends=True),
        }
    )

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
        "title": title,
        "code_file": "job.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": accelerator,
        "dataset_sources": [dataset_ref] if dataset_ref else [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

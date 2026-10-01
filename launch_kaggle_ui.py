from __future__ import annotations

import os
import importlib.util
import subprocess
import sys
from pathlib import Path

def load_local_env() -> None:
    path = Path(os.getenv("QWEN_KAGGLE_ENV_FILE", Path(__file__).resolve().parent / ".env.local"))
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())

load_local_env()

def ensure_dependencies() -> None:
    missing = []
    if importlib.util.find_spec("gradio") is None:
        missing.append("gradio>=5.0")
    if shutil.which("kaggle") is None and importlib.util.find_spec("kaggle") is None:
        missing.append("kaggle>=1.7")
    if missing:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing],
            check=True,
        )

import shutil
ensure_dependencies()

from kaggle_app.config import SETTINGS
import gradio as gr
from kaggle_app.ui import build_ui, CSS

if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=8)
    demo.launch(
        share=SETTINGS.share_gradio,
        inline=False,
        show_error=True,
        prevent_thread_lock=False,
        allowed_paths=[str(SETTINGS.storage_root)],
        theme=gr.themes.Soft(),
        css=CSS,
    )

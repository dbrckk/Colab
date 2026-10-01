from __future__ import annotations

import os
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import subprocess
import sys
import secrets
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

def _version_tuple(package: str) -> tuple[int, int, int]:
    try:
        raw = version(package)
        nums = [int(x) for x in __import__("re").findall(r"\d+", raw)[:3]]
        return tuple((nums + [0, 0, 0])[:3])
    except (PackageNotFoundError, ValueError):
        return (0, 0, 0)

def ensure_dependencies() -> None:
    missing = []
    if _version_tuple("gradio") < (6, 0, 0):
        missing.append("gradio>=6.0")
    if _version_tuple("kaggle") < (2, 2, 3) or shutil.which("kaggle") is None:
        missing.append("kaggle>=2.2.3")
    if missing:
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--disable-pip-version-check", "--upgrade", *missing
            ],
            check=True,
        )

import shutil
ensure_dependencies()

from kaggle_app.config import SETTINGS
import gradio as gr
from kaggle_app.ui import build_ui, CSS

if __name__ == "__main__":
    demo = build_ui()
    ui_user = os.getenv("QWEN_UI_USER", "qwen")
    ui_password = os.getenv("QWEN_UI_PASSWORD") or secrets.token_urlsafe(10)
    if SETTINGS.share_gradio:
        print("🔐 Connexion Gradio")
        print("Utilisateur :", ui_user)
        print("Mot de passe :", ui_password)
    demo.queue(default_concurrency_limit=8)
    demo.launch(
        share=SETTINGS.share_gradio,
        inline=False,
        show_error=True,
        prevent_thread_lock=False,
        allowed_paths=[str(SETTINGS.storage_root)],
        theme=gr.themes.Soft(),
        css=CSS,
        auth=(ui_user, ui_password) if SETTINGS.share_gradio else None,
    )

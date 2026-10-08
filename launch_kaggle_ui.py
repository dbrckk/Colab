from __future__ import annotations

import os
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import subprocess
import sys
import secrets
import shutil
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
    gradio_version = _version_tuple("gradio")
    kaggle_version = _version_tuple("kaggle")
    if not ((6, 0, 0) <= gradio_version < (7, 0, 0)):
        missing.append("gradio>=6.0,<7.0")
    if not ((2, 2, 3) <= kaggle_version < (3, 0, 0)) or shutil.which("kaggle") is None:
        missing.append("kaggle>=2.2.3,<3.0")
    if missing:
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--disable-pip-version-check", "--upgrade", *missing
            ],
            check=True,
        )

ensure_dependencies()

from kaggle_app.config import SETTINGS
import gradio as gr
from kaggle_app.ui import build_ui, CSS

if __name__ == "__main__":
    demo = build_ui()
    ui_user = os.getenv("QWEN_UI_USER", "qwen")
    ui_password = os.getenv("QWEN_UI_PASSWORD") or secrets.token_urlsafe(10)
    hosted = bool(os.getenv("PORT"))
    if hosted and not os.getenv("QWEN_UI_PASSWORD"):
        raise RuntimeError("QWEN_UI_PASSWORD est obligatoire pour un site public.")
    if SETTINGS.share_gradio or hosted:
        print("🔐 Authentification Gradio activée.")
        print("Utilisateur :", ui_user)
        if not hosted:
            # Temporary Colab share links use a generated password; show it
            # only there, never leak a hosted service secret into build logs.
            print("Mot de passe :", ui_password)
    demo.queue(default_concurrency_limit=8)
    demo.launch(
        share=SETTINGS.share_gradio and not hosted,
        server_name="0.0.0.0" if hosted else None,
        server_port=int(os.getenv("PORT", "7860")),
        inline=False,
        show_error=True,
        prevent_thread_lock=False,
        allowed_paths=[str(SETTINGS.storage_root)],
        theme=gr.themes.Soft(),
        css=CSS,
        auth=(ui_user, ui_password) if (SETTINGS.share_gradio or hosted) else None,
    )

from __future__ import annotations

import os
from pathlib import Path

def load_local_env() -> None:
    path = Path(__file__).resolve().parent / ".env.local"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())

load_local_env()

from kaggle_app.config import SETTINGS
from kaggle_app.ui import build_ui

if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=8)
    demo.launch(
        share=SETTINGS.share_gradio,
        inline=False,
        show_error=True,
        prevent_thread_lock=False,
    )

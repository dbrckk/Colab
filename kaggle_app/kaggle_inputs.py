from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ALLOWED_TASKS = {"image", "image_edit", "video_faceswap"}
ALLOWED_ASPECTS = {"1:1", "4:3", "3:4", "16:9", "9:16"}


def validate_submission(
    task: str,
    prompt: str,
    steps: int,
    cfg: float,
    seed: int,
    aspect: str,
    source_image: str | None = None,
    target_video: str | None = None,
) -> None:
    if task not in ALLOWED_TASKS:
        raise ValueError(f"Tâche non supportée: {task}")
    if task in {"image", "image_edit"} and not (prompt or "").strip():
        raise ValueError("Le prompt est vide.")
    if not 1 <= int(steps) <= 100:
        raise ValueError("Steps doit être compris entre 1 et 100.")
    if not 0.0 <= float(cfg) <= 30.0:
        raise ValueError("CFG doit être compris entre 0 et 30.")
    if int(seed) < -1:
        raise ValueError("Seed doit valoir -1 (aléatoire) ou être positive.")
    if aspect not in ALLOWED_ASPECTS:
        raise ValueError(f"Format non supporté: {aspect}")

    if task == "image_edit" and not source_image:
        raise ValueError("image_edit requiert une image source.")
    if task == "video_faceswap" and (not source_image or not target_video):
        raise ValueError("video_faceswap requiert un visage source et une vidéo cible.")

    if source_image:
        p = Path(source_image)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Image source introuvable: {p}")
        try:
            from PIL import Image

            with Image.open(p) as img:
                img.verify()
        except Exception as exc:
            raise ValueError(f"Image source illisible ou invalide: {p.name}") from exc

    if target_video:
        p = Path(target_video)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Vidéo cible introuvable: {p}")
        if p.stat().st_size <= 0:
            raise ValueError("La vidéo cible est vide.")

        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            probe = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "default=nokey=1:noprint_wrappers=1",
                    str(p),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if probe.returncode != 0 or "video" not in (probe.stdout or "").lower():
                raise ValueError(
                    f"Vidéo cible illisible ou sans piste vidéo: {p.name}"
                )

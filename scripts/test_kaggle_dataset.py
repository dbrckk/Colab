import json
import tempfile
from pathlib import Path

from kaggle_app.kaggle_dataset import write_dataset_bundle


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    source = root / "portrait.webp"
    video = root / "clip.mp4"
    source.write_bytes(b"image")
    video.write_bytes(b"video")

    bundle = root / "bundle"
    bundle.mkdir()
    staged = write_dataset_bundle(
        bundle,
        job_id="job123",
        dataset_ref="user/qwen-input-job123",
        config={
            "task": "video_faceswap",
            "source_image": str(source),
            "target_video": str(video),
            "prompt": "",
        },
    )

    assert staged["source_image"] == "source_image.webp"
    assert staged["target_video"] == "target_video.mp4"
    assert (bundle / "source_image.webp").read_bytes() == b"image"
    assert (bundle / "target_video.mp4").read_bytes() == b"video"

    saved_config = json.loads((bundle / "job_config.json").read_text(encoding="utf-8"))
    assert saved_config["source_image"] == "source_image.webp"
    assert saved_config["target_video"] == "target_video.mp4"

    metadata = json.loads((bundle / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "user/qwen-input-job123"
    assert metadata["title"] == "Qwen input job123"

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    bundle = root / "bundle"
    bundle.mkdir()
    missing = root / "missing.png"
    try:
        write_dataset_bundle(
            bundle,
            job_id="missing",
            dataset_ref="user/missing",
            config={"source_image": str(missing)},
        )
        raise AssertionError("missing source file accepted")
    except FileNotFoundError:
        pass

print("Kaggle dataset bundle staging tests passed.")

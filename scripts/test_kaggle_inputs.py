import tempfile
from pathlib import Path

from kaggle_app.kaggle_inputs import validate_submission


validate_submission("image", "hello", 25, 1.0, -1, "1:1")

invalid = [
    ("unknown", "x", 25, 1.0, -1, "1:1", None, None),
    ("image", "", 25, 1.0, -1, "1:1", None, None),
    ("image", "x", 0, 1.0, -1, "1:1", None, None),
    ("image", "x", 101, 1.0, -1, "1:1", None, None),
    ("image", "x", 25, -0.1, -1, "1:1", None, None),
    ("image", "x", 25, 31.0, -1, "1:1", None, None),
    ("image", "x", 25, 1.0, -2, "1:1", None, None),
    ("image", "x", 25, 1.0, -1, "2:1", None, None),
    ("image_edit", "edit", 25, 1.0, -1, "1:1", None, None),
    ("video_faceswap", "", 25, 1.0, -1, "1:1", None, None),
]
for args in invalid:
    try:
        validate_submission(*args)
        raise AssertionError(f"invalid submission accepted: {args!r}")
    except ValueError:
        pass

with tempfile.TemporaryDirectory() as td:
    missing = Path(td) / "missing.png"
    try:
        validate_submission("image_edit", "edit", 25, 1.0, -1, "1:1", str(missing))
        raise AssertionError("missing source image accepted")
    except FileNotFoundError:
        pass

with tempfile.TemporaryDirectory() as td:
    bad = Path(td) / "bad.png"
    bad.write_bytes(b"not-an-image")
    try:
        validate_submission("image_edit", "edit", 25, 1.0, -1, "1:1", str(bad))
        raise AssertionError("invalid source image accepted")
    except ValueError:
        pass

with tempfile.TemporaryDirectory() as td:
    empty_video = Path(td) / "empty.mp4"
    empty_video.write_bytes(b"")
    source = Path(td) / "source.png"
    try:
        from PIL import Image
        Image.new("RGB", (1, 1)).save(source)
    except Exception:
        source.write_bytes(b"")
    try:
        validate_submission(
            "video_faceswap",
            "",
            25,
            1.0,
            -1,
            "1:1",
            str(source),
            str(empty_video),
        )
        raise AssertionError("empty target video accepted")
    except ValueError:
        pass

print("Kaggle submission validation tests passed.")

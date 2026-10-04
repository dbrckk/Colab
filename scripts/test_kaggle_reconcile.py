import tempfile
from pathlib import Path

from kaggle_app.kaggle_reconcile import (
    classify_missing_local_outputs,
    has_valid_local_media,
)


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    image = root / "image.png"
    image.write_bytes(b"x")

    assert has_valid_local_media([
        {"kind": "image", "path": str(image)},
    ]) is True
    assert has_valid_local_media([
        {"kind": "video", "path": str(root / "missing.mp4")},
    ]) is False
    assert has_valid_local_media([
        {"kind": "file", "path": str(image)},
    ]) is False

assert classify_missing_local_outputs(
    kernel_ref="",
    kernel_state="unknown",
)[0:2] == ("no_remote", False)

outcome, can_recover, error = classify_missing_local_outputs(
    kernel_ref="user/kernel",
    kernel_state="complete",
)
assert outcome == "recoverable"
assert can_recover is True
assert "encore disponibles" in error

outcome, can_recover, error = classify_missing_local_outputs(
    kernel_ref="user/kernel",
    kernel_state="error",
)
assert outcome == "remote_failed"
assert can_recover is False
assert "plus récupérable" in error

outcome, can_recover, error = classify_missing_local_outputs(
    kernel_ref="user/kernel",
    kernel_state="unknown",
)
assert outcome == "unknown"
assert can_recover is False
assert "n'a pas pu être confirmé" in error

print("Kaggle completed-job reconciliation policy tests passed.")

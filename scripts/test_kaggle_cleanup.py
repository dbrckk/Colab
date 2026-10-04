import os
import tempfile
import time
from pathlib import Path

from kaggle_app.kaggle_cleanup import (
    cleanup_report,
    expired_exports,
    format_reclaimed_bytes,
    orphan_input_dirs,
)


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    exports = root / "exports"
    exports.mkdir()
    old = exports / "old.zip"
    fresh = exports / "fresh.zip"
    old.write_bytes(b"old")
    fresh.write_bytes(b"fresh")

    now = time.time()
    old_ts = now - 20 * 86400
    os.utime(old, (old_ts, old_ts))
    os.utime(fresh, (now, now))

    result = expired_exports(exports, now=now, max_age_days=14)
    assert result == [old]

    inputs = root / "inputs"
    inputs.mkdir()
    (inputs / "known").mkdir()
    (inputs / "orphan").mkdir()
    (inputs / "file.txt").write_text("x", encoding="utf-8")
    orphaned = orphan_input_dirs(inputs, known_ids={"known"})
    assert [p.name for p in orphaned] == ["orphan"]

assert format_reclaimed_bytes(0) == "0.0 Mo"
assert format_reclaimed_bytes(1024**3) == "1.00 Go"

report = cleanup_report(
    removed_exports=1,
    removed_inputs=2,
    gc_files=3,
    stale_artifacts=4,
    recoverable_jobs=5,
    expired_recovery_kernels=6,
    reclaimed=1024**3,
)
assert "1 export(s) ancien(s)" in report
assert "2 dossier(s) d'entrée orphelin(s)" in report
assert "6 récupération(s) distante(s) expirée(s)" in report
assert "1.00 Go libéré(s)" in report

print("Kaggle cleanup helper tests passed.")

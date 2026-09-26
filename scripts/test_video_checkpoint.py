from __future__ import annotations

import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT_REPO = Path(__file__).resolve().parents[1]
MODULE = ROOT_REPO / "qwen_studio_v8" / "05b_video_faceswap.py"

with tempfile.TemporaryDirectory() as td:
    base = Path(td)
    root = base / "runtime"
    jobs = base / "jobs"
    outputs = base / "outputs"
    root.mkdir(); jobs.mkdir(); outputs.mkdir()

    state = {}
    def load_job(job_id):
        return state.get(str(job_id))
    def save_job(job):
        state[str(job["id"])] = dict(job)
        return job
    def list_jobs(limit=50):
        return list(state.values())[:limit]

    ns = {
        "ROOT": str(root),
        "JOB_ROOT": str(jobs),
        "OUTPUT_ROOT": str(outputs),
        "USE_DRIVE": False,
        "DRIVE_ROOT": str(base / "drive"),
        "QWEN_RUNTIME_ID": "ci-runtime",
        "VIDEO_FACE_SWAP_PERSIST_CHECKPOINTS": True,
        "VIDEO_FACE_SWAP_DELETE_CHECKPOINTED_FRAMES": True,
        "VIDEO_FACE_SWAP_PAUSE_ENABLED": True,
        "VIDEO_FACE_SWAP_MIN_FREE_DISK_GB": 0,
        "VIDEO_FACE_SWAP_CACHE_MAX_AGE_HOURS": 24,
        "VIDEO_FACE_SWAP_CACHE_KEEP_RECENT": 3,
        "VIDEO_FACEFUSION_VERSION": "3.9.0",
        "VIDEO_FACEFUSION_ROOT": str(base / "facefusion"),
        "VIDEO_FACEFUSION_VENV": str(base / "facefusion-venv"),
        "VIDEO_FACEFUSION_CACHE_ENABLED": False,
        "VIDEO_FACEFUSION_CACHE_DIR": str(base / "ff-cache"),
        "VIDEO_FACEFUSION_CACHE_TAR": str(base / "ff-cache.tar.gz"),
        "load_job": load_job,
        "save_job": save_job,
        "list_jobs": list_jobs,
        "_execute_job": lambda job_id: None,
        "JOB_LOCK": threading.Lock(),
        "JOB_FUTURES": {},
        "JOB_EXECUTOR": ThreadPoolExecutor(max_workers=1),
    }

    exec(compile(MODULE.read_text(encoding="utf-8"), str(MODULE), "exec"), ns)

    frames_out = base / "frames"
    frames_out.mkdir()
    for idx in range(1, 4):
        # Checkpoint logic only needs files; real PNG decoding happens elsewhere.
        (frames_out / f"{idx:08d}.png").write_bytes((b"frame-%d-" % idx) + b"x" * 2048)

    ns["_checkpoint_frames"]("job1", str(frames_out), 1, 3)

    checkpoint_dir = Path(ns["VIDEO_CHECKPOINT_ROOT"]) / "job1"
    archive = checkpoint_dir / "chunk_00000001_00000003.tar"
    manifest = Path(str(archive) + ".json")

    assert archive.exists(), "checkpoint archive missing"
    assert manifest.exists(), "checkpoint manifest missing"
    assert ns["_checkpoint_is_valid"](archive), "fresh checkpoint should validate"
    assert ns["_latest_checkpoint_end"]("job1") == 3
    assert ns["_restore_video_checkpoints"]("job1", str(frames_out)) == 3
    assert not list(frames_out.glob("*.png")), "checkpointed frames should be deleted locally"

    # Checksum manifest must detect corruption.
    with archive.open("ab") as fh:
        fh.write(b"corruption")
    assert not ns["_checkpoint_is_valid"](archive), "corrupted checkpoint was accepted"

    # A gap must stop resume at the last contiguous frame.
    gap_frames = base / "gap_frames"
    gap_frames.mkdir()
    for idx in (1, 2, 4, 5):
        (gap_frames / f"{idx:08d}.png").write_bytes((b"gap-%d-" % idx) + b"y" * 2048)
    ns["_checkpoint_frames"]("job2", str(gap_frames), 1, 2)
    ns["_checkpoint_frames"]("job2", str(gap_frames), 4, 5)
    assert ns["_latest_checkpoint_end"]("job2") == 2, "resume crossed a checkpoint gap"
    segments = ns["_contiguous_checkpoint_segments"]("job2")
    assert segments and segments[-1][2] == 2

    # Preflight quality parser.
    ratio = ns["_preflight_protected_ratio"](
        "swap_nouveaux=92 | protégées_nouvelles=8"
    )
    assert abs(ratio - 0.08) < 1e-9
    assert ns["_preflight_protected_ratio"]("no metrics") is None

    # Automatic backend resolver must respect explicit choice and analysis recommendation.
    ns["VIDEO_FACEFUSION_ENABLED"] = True
    ns["VIDEO_FACEFUSION_AUTO_INSTALL_ON_HARD"] = True
    ns["analyze_video_difficulty"] = lambda _: ("hard", "facefusion-ultra")
    ns["_facefusion_ready"] = lambda: False
    assert ns["_resolve_video_backend"]("dummy.mp4", "builtin-ultra") == "builtin-ultra"
    assert ns["_resolve_video_backend"]("dummy.mp4", "facefusion-ultra") == "facefusion-ultra"
    assert ns["_resolve_video_backend"]("dummy.mp4", "auto") == "facefusion-ultra"

    # UI status exposes backend and preflight quality decisions.
    status = ns["_format_video_status"]({
        "status": "running",
        "updated_at": "now",
        "resolved_backend": "facefusion-ultra",
        "preflight_protected_ratio": 0.125,
        "progress_total": 100,
        "progress_current": 25,
        "progress_pct": 25.0,
        "error": "",
    })
    assert "Backend: facefusion-ultra" in status
    assert "Préflight protégé: 12.5%" in status
    assert "25/100" in status

    ns["JOB_EXECUTOR"].shutdown(wait=False)

print("Video checkpoint runtime test passed.")

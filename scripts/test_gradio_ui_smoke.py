from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "qwen_studio_v8"

ui = "".join(
    (MOD / name).read_text(encoding="utf-8")
    for name in ["06_ui.part00", "06_ui.part01", "06_ui.part02"]
)

def job():
    return {"id": "smoke-job"}

ns = {
    "GRADIO_ANALYTICS": False,
    "BOOTSTRAP_STATE": {
        "models_ready": False,
        "runtime_ready": False,
        "phase": "idle",
        "details": "",
        "last_error": "",
    },
    "start_background_prepare": lambda force=False: "ok",
    "jobs_table": lambda *args, **kwargs: [],
    "submit_generate_job": lambda *args, **kwargs: job(),
    "submit_edit_job": lambda *args, **kwargs: job(),
    "submit_faceswap_job": lambda *args, **kwargs: job(),
    "submit_faceswap_video_job": lambda *args, **kwargs: job(),
    "cancel_job": lambda *_: "cancelled",
    "pause_job": lambda *_: "paused",
    "resume_job": lambda *_: "running",
    "analyze_video_difficulty": lambda *_: ("Analyse smoke test", "builtin-ultra"),
    "cleanup_video_cache": lambda *_: "clean",
    "facefusion_status_text": lambda: "FaceFusion smoke",
    "video_selftest_report": lambda: "self-test smoke",
    "restart_video_job": lambda *_: "restarted",
    "job_status_video_view": lambda *_: ("idle", None, ""),
    "job_status_view": lambda *_: ("idle", None, ""),
    "latest_image": lambda: (None, ""),
    "bootstrap_status_text": lambda: "bootstrap smoke",
    "ASPECTS": {"1:1": (1024, 1024), "16:9": (1344, 768)},
    "UI_TITLE": "Qwen Studio Smoke",
    "UI_SUBTITLE": "UI smoke test",
    "VIDEO_FACE_SWAP_PREVIEW_SECONDS": 5,
    "VIDEO_FACE_SWAP_FRAME_STRIDE": 1,
    "VIDEO_FACE_SWAP_MAX_FRAMES": 0,
    "VIDEO_FACE_SWAP_PREFLIGHT_SECONDS": 5,
    "VIDEO_FACE_SWAP_KEEP_AUDIO": True,
    "VIDEO_FACE_SWAP_CRF": 17,
    "VIDEO_FACE_SWAP_PRESET": "slow",
    "VIDEO_FACE_SWAP_TARGET_FACE_POSITION": 0,
    "VIDEO_FACE_SWAP_REFERENCE_FRAME": 0,
    "VIDEO_FACE_SWAP_REFERENCE_DISTANCE": 0.30,
    "VIDEO_FACE_SWAP_AUTO_REFRESH": True,
    "AUTO_REFRESH_SECONDS": 3,
    "HERETIC_PATH": "/tmp/heretic.gguf",
    "MMPROJ_PATH": "/tmp/mmproj.gguf",
    "DIT_PATH": "/tmp/dit.gguf",
    "VAE_PATH": "/tmp/vae.safetensors",
}

compiled = compile(ui, "qwen_studio_ui_smoke.py", "exec")
exec(compiled, ns)

assert "demo" in ns, "Gradio Blocks object was not created"
assert ns["demo"] is not None
print("Gradio UI smoke test passed.")

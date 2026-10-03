from kaggle_app import ui

demo = ui.build_ui()
assert demo is not None

original_job = ui.controller.job
try:
    ui.controller.job = lambda _job_id: {
        "id": "batch-ui-test",
        "task": "image_batch",
        "prompt": "Lot de 2 images",
        "meta": {
            "prompts": ["first prompt", "second prompt"],
            "negative_prompt": "bad",
            "steps": 22,
            "cfg": 2.0,
            "seed": 10,
            "aspect": "16:9",
        },
    }
    loaded = ui._load_job_to_form("batch-ui-test")
    assert loaded[0] == "image"
    assert loaded[1] == "first prompt"
    assert loaded[9] == "first prompt\nsecond prompt"
    assert loaded[3] == 22
    assert loaded[6] == "16:9"
finally:
    ui.controller.job = original_job

print("Kaggle Studio UI smoke test passed.")


assert "Authentification Kaggle" in ui._friendly_error("403 Forbidden")
assert "Limite/quota Kaggle" in ui._friendly_error("429 Too Many Requests")
assert "Timeout" in ui._friendly_error("kernel timeout")
assert "Entrée invalide" in ui._friendly_error("Image source illisible")
assert "Worker Kaggle" in ui._friendly_error("result.json illisible")
print("friendly error classification passed.")


original_job = ui.controller.job
original_artifacts = ui.controller.artifacts
original_jobs = ui.controller.jobs
original_queue_position = ui.controller.queue_position
original_job_storage_bytes = ui.controller.job_storage_bytes
try:
    ui.controller.job = lambda _job_id: {
        "id": "wait-auth-remote",
        "task": "image",
        "status": "waiting_auth",
        "kernel_ref": "ci-user/existing-kernel",
        "prompt": "x",
        "error": "Authentication required",
        "meta": {"recover_outputs_available": True, "auto_recovery_attempts": 1},
        "created_at": 1,
        "updated_at": 1,
    }
    ui.controller.artifacts = lambda _job_id: []
    ui.controller.jobs = lambda _limit=100: []
    ui.controller.queue_position = lambda _job_id: 1
    ui.controller.job_storage_bytes = lambda _job_id: 0
    status, *_ = ui._refresh("wait-auth-remote")
    assert "kernel distant est conservé" in status
    assert "reprendre sans recalcul" in status

    ui.controller.job = lambda _job_id: {
        "id": "wait-auth-local",
        "task": "image",
        "status": "waiting_auth",
        "kernel_ref": "",
        "prompt": "x",
        "error": "Authentication required",
        "meta": {},
        "created_at": 1,
        "updated_at": 1,
    }
    status, *_ = ui._refresh("wait-auth-local")
    assert "préparation locale est conservée" in status
    assert "reprendre automatiquement" in status
finally:
    ui.controller.job = original_job
    ui.controller.artifacts = original_artifacts
    ui.controller.jobs = original_jobs
    ui.controller.queue_position = original_queue_position
    ui.controller.job_storage_bytes = original_job_storage_bytes

print("Waiting-auth UI guidance passed.")

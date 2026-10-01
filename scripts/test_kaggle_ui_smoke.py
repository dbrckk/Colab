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

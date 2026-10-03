from kaggle_app.kaggle_jobs import (
    build_job_config,
    dataset_ref,
    kernel_ref,
    needs_dataset,
    slugify,
)


assert slugify("Hello WORLD !!!") == "hello-world"
assert slugify("---") == "qwen-job"
assert len(slugify("x" * 100)) == 48

job = {
    "id": "abc123",
    "task": "image",
    "prompt": "hello",
    "meta": {"seed": 42, "aspect": "1:1"},
}
config = build_job_config(job)
assert config["job_id"] == "abc123"
assert config["task"] == "image"
assert config["prompt"] == "hello"
assert config["seed"] == 42

assert needs_dataset(job) is False
assert needs_dataset({
    "id": "edit",
    "task": "image_edit",
    "meta": {"source_image": "/tmp/source.png"},
}) is True
assert needs_dataset({
    "id": "video",
    "task": "video_faceswap",
    "meta": {"target_video": "/tmp/video.mp4"},
}) is True

assert dataset_ref("User.Name", "ABC 123") == "User.Name/qwen-input-abc-123"
ref, slug = kernel_ref("User.Name", "ABC 123")
assert ref == "User.Name/qwen-studio-abc-123"
assert slug == "qwen-studio-abc-123"

print("Kaggle job naming and config tests passed.")

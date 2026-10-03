import json
import tempfile
from pathlib import Path

from kaggle_app.kaggle_kernel import write_kernel_bundle


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    write_kernel_bundle(
        root,
        kernel_ref="user/qwen-studio-job1",
        title="qwen-studio-job1",
        worker_source="print('worker')\n",
        job_config={"task": "image", "prompt": "hello"},
        dataset_ref="",
        accelerator="NvidiaTeslaT4",
    )

    notebook = json.loads((root / "job.ipynb").read_text(encoding="utf-8"))
    metadata = json.loads((root / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert len(notebook["cells"]) == 3
    assert notebook["cells"][1]["cell_type"] == "code"
    bootstrap = "".join(notebook["cells"][1]["source"])
    assert "job_config.json" in bootstrap
    assert "hello" in bootstrap
    assert "print('worker')" in "".join(notebook["cells"][2]["source"])

    assert metadata["id"] == "user/qwen-studio-job1"
    assert metadata["dataset_sources"] == []
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    write_kernel_bundle(
        root,
        kernel_ref="user/qwen-studio-job2",
        title="qwen-studio-job2",
        worker_source="print('worker')\n",
        job_config={"task": "image_edit"},
        dataset_ref="user/qwen-input-job2",
        accelerator="NvidiaTeslaT4",
    )

    notebook = json.loads((root / "job.ipynb").read_text(encoding="utf-8"))
    metadata = json.loads((root / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert len(notebook["cells"]) == 2
    assert metadata["dataset_sources"] == ["user/qwen-input-job2"]
    assert "job_config.json" not in "".join(notebook["cells"][1]["source"])

print("Kaggle kernel bundle generation tests passed.")

from __future__ import annotations

import subprocess

def help_text(*args: str) -> str:
    p = subprocess.run(["kaggle", *args, "--help"], capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise RuntimeError((p.stdout or "") + "\n" + (p.stderr or ""))
    return (p.stdout or "") + "\n" + (p.stderr or "")

kernels_push = help_text("kernels", "push")
for flag in ("--path", "--timeout", "--accelerator"):
    assert flag in kernels_push, f"Missing kernels push flag: {flag}"

kernels_output = help_text("kernels", "output")
for flag in ("--path", "--force", "--quiet"):
    assert flag in kernels_output, f"Missing kernels output flag: {flag}"

kernels_delete = help_text("kernels", "delete")
assert "--yes" in kernels_delete

datasets_create = help_text("datasets", "create")
for flag in ("--path", "--quiet", "--keep-tabular", "--dir-mode"):
    assert flag in datasets_create, f"Missing datasets create flag: {flag}"

datasets_status = help_text("datasets", "status")
assert "DATASET" in datasets_status.upper()

datasets_delete = help_text("datasets", "delete")
assert "--yes" in datasets_delete

print("Kaggle CLI contract test passed.")

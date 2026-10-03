from kaggle_app.kaggle_protocol import (
    is_auth_cli_error,
    is_transient_cli_error,
    parse_dataset_status,
    parse_kernel_status,
    remote_kernel_missing,
)


assert is_auth_cli_error("401 Unauthorized")
assert is_auth_cli_error("Authentication required to call the Kaggle API.")
assert not is_auth_cli_error("temporary network timeout")

assert is_transient_cli_error("502 Bad Gateway")
assert is_transient_cli_error("429 Too Many Requests")
assert not is_transient_cli_error("403 Forbidden")

assert remote_kernel_missing("404 kernel not found")
assert remote_kernel_missing("No such kernel")
assert not remote_kernel_missing("503 Service Unavailable")

dataset_cases = {
    "status: ready": "ready",
    "Dataset Status = completed": "ready",
    "pending": "pending",
    "FAILED": "error",
}
for raw, expected in dataset_cases.items():
    assert parse_dataset_status(raw) == expected

for raw in (
    "pending\nlast completed version: 2",
    "status unknown; no error detected",
    "ready cache metadata\ncreating now",
):
    assert parse_dataset_status(raw) == "unknown"

kernel_cases = {
    "status: running": "running",
    "Kernel Status = completed": "complete",
    "queued": "queued",
    "FAILED": "error",
}
for raw, expected in kernel_cases.items():
    assert parse_kernel_status(raw) == expected

for raw in (
    "running\nlast completed version: 4",
    "status unknown; no error detected",
    "completed build metadata\nrunning now",
):
    try:
        parse_kernel_status(raw)
        raise AssertionError(f"ambiguous kernel status unexpectedly accepted: {raw}")
    except RuntimeError as exc:
        assert (
            "Statut Kaggle ambigu" in str(exc)
            or "Statut Kaggle non reconnu" in str(exc)
        )

print("Kaggle protocol helper tests passed.")

from kaggle_app.kaggle_job_policy import (
    can_cancel_locally,
    delete_block_reason,
    retry_block_reason,
    terminal_status_message,
)


assert terminal_status_message({"status": "done"}) == "Job déjà terminé: done"
assert terminal_status_message({"status": "error"}) == "Job déjà terminé: error"
assert terminal_status_message({"status": "running"}) is None

assert retry_block_reason({"status": "running"}) == "Ce job est encore actif."
assert "kernel Kaggle existant" in retry_block_reason({
    "status": "waiting_auth",
    "kernel_ref": "user/kernel",
})
assert "file locale" in retry_block_reason({
    "status": "waiting_auth",
    "kernel_ref": "",
})
assert retry_block_reason({"status": "error"}) is None
assert retry_block_reason({"status": "done"}) is None

assert can_cancel_locally(
    kernel_ref="",
    future_active=False,
    cancelled_before_start=False,
) is True
assert can_cancel_locally(
    kernel_ref="",
    future_active=True,
    cancelled_before_start=True,
) is True
assert can_cancel_locally(
    kernel_ref="",
    future_active=True,
    cancelled_before_start=False,
) is False
assert can_cancel_locally(
    kernel_ref="user/kernel",
    future_active=False,
    cancelled_before_start=False,
) is False

print("Kaggle job state policy tests passed.")


for status in (
    "queued",
    "running",
    "submitting",
    "recovering",
    "downloading",
    "cancel_requested",
):
    assert delete_block_reason({"status": status}) == "Annule d'abord le job actif."

assert delete_block_reason({"status": "done"}) is None
assert delete_block_reason({"status": "error"}) is None
assert delete_block_reason({"status": "waiting_auth"}) is None

print("Kaggle local deletion policy tests passed.")

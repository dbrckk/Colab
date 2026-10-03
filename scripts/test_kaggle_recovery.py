from kaggle_app.kaggle_recovery import (
    MAX_AUTO_RECOVERY_ATTEMPTS,
    auto_recovery_attempts,
    keep_remote_kernel_for_retry,
    recoverable_output_candidate,
    startup_recovery_action,
)


def row(status, kernel_ref="", meta=None):
    return {
        "id": "job",
        "status": status,
        "kernel_ref": kernel_ref,
        "meta": meta or {},
    }


assert startup_recovery_action(row("preparing"), True) == "resume_local"
assert startup_recovery_action(row("preparing"), False) == "wait_local_auth"
assert startup_recovery_action(row("uploading_inputs"), True) == "resume_local"

assert startup_recovery_action(row("running", "u/k"), True) == "resume_remote"
assert startup_recovery_action(row("running", "u/k"), False) == "wait_remote_auth"
assert startup_recovery_action(row("submitting"), True) == "resume_local"

assert startup_recovery_action(row("queued"), True) == "resume_local"
assert startup_recovery_action(row("queued"), False) == "wait_local_auth"
assert startup_recovery_action(row("waiting_auth", "u/k"), True) == "resume_remote"
assert startup_recovery_action(row("waiting_auth", "u/k"), False) == "wait_remote_auth"

recoverable = row(
    "error",
    "u/k",
    {"recover_outputs_available": True, "auto_recovery_attempts": 2},
)
assert startup_recovery_action(recoverable, True) == "resume_recoverable"
assert startup_recovery_action(recoverable, False) == "none"
assert recoverable_output_candidate(recoverable) is True
assert auto_recovery_attempts(recoverable) == 2

exhausted = row(
    "error",
    "u/k",
    {
        "recover_outputs_available": True,
        "auto_recovery_attempts": MAX_AUTO_RECOVERY_ATTEMPTS,
    },
)
assert startup_recovery_action(exhausted, True) == "none"
assert recoverable_output_candidate(exhausted) is False

assert recoverable_output_candidate(
    row("error", "", {"recover_outputs_available": True})
) is False
assert recoverable_output_candidate(
    row("done", "u/k", {"recover_outputs_available": True})
) is False

assert keep_remote_kernel_for_retry(
    "waiting_auth",
    {"recover_outputs_available": True},
    "u/k",
) is True
assert keep_remote_kernel_for_retry(
    "error",
    {"recover_outputs_available": True},
    "u/k",
) is True
assert keep_remote_kernel_for_retry(
    "done",
    {"recover_outputs_available": True},
    "u/k",
) is False
assert keep_remote_kernel_for_retry(
    "error",
    {"recover_outputs_available": False},
    "u/k",
) is False

print("Kaggle recovery policy tests passed.")

from kaggle_app.kaggle_recovery import (
    MAX_AUTO_RECOVERY_ATTEMPTS,
    auto_recovery_attempts,
    classify_remote_recovery_failure,
    has_recoverable_outputs,
    keep_remote_kernel_for_retry,
    preserve_kernel_after_execute_failure,
    recoverable_output_candidate,
    recovery_marked,
    remote_kernel_is_preserved,
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


assert has_recoverable_outputs(
    row("error", "u/k", {"recover_outputs_available": True})
) is True
assert has_recoverable_outputs(
    row("error", "", {"recover_outputs_available": True})
) is False

assert remote_kernel_is_preserved(
    row("waiting_auth", "u/k", {})
) is True
assert remote_kernel_is_preserved(
    row("recovering", "u/k", {})
) is True
assert remote_kernel_is_preserved(
    row("error", "u/k", {"recover_outputs_available": True})
) is True
assert remote_kernel_is_preserved(
    row("error", "u/k", {"recover_outputs_available": False})
) is False

print("Kaggle recovery availability predicate tests passed.")


marked_without_kernel = row(
    "error",
    "",
    {"recover_outputs_available": True},
)
assert recovery_marked(marked_without_kernel) is True
assert has_recoverable_outputs(marked_without_kernel) is False

print("Kaggle recovery marker semantics passed.")


assert classify_remote_recovery_failure(
    auth_required=True,
    missing_remote=False,
    remote_failed=False,
) == ("waiting_auth", True, "auth_required", False)

assert classify_remote_recovery_failure(
    auth_required=False,
    missing_remote=True,
    remote_failed=False,
) == ("error", False, "remote_missing", False)

assert classify_remote_recovery_failure(
    auth_required=False,
    missing_remote=False,
    remote_failed=True,
) == ("error", False, "remote_failed", True)

assert classify_remote_recovery_failure(
    auth_required=False,
    missing_remote=False,
    remote_failed=False,
) == ("error", True, "downloading", False)

assert preserve_kernel_after_execute_failure(
    kernel_ref="u/k",
    failed_phase="running",
    remote_failed=False,
) is True
assert preserve_kernel_after_execute_failure(
    kernel_ref="u/k",
    failed_phase="preparing",
    remote_failed=False,
) is False
assert preserve_kernel_after_execute_failure(
    kernel_ref="u/k",
    failed_phase="running",
    remote_failed=True,
) is False

print("Kaggle recovery failure classification tests passed.")

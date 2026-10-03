from kaggle_app.kaggle_monitor import wait_for_kernel


states = iter([
    ("queued", "status: queued"),
    ("running", "status: running"),
    ("complete", "status: complete"),
])
seen = []
clock = {"t": 0.0}

result = wait_for_kernel(
    "user/kernel",
    kernel_timeout=60,
    poll_seconds=1,
    check_cancelled=lambda: None,
    kernel_status=lambda _ref: next(states),
    on_status=lambda state, raw: seen.append((state, raw)),
    fetch_logs=lambda _ref: "unused",
    failure_factory=lambda logs: RuntimeError(logs),
    timeout_message="timeout",
    now=lambda: clock["t"],
    sleep=lambda seconds: clock.__setitem__("t", clock["t"] + seconds),
)
assert result == "status: complete"
assert [state for state, _ in seen] == ["queued", "running", "complete"]

try:
    wait_for_kernel(
        "user/kernel",
        kernel_timeout=60,
        poll_seconds=1,
        check_cancelled=lambda: None,
        kernel_status=lambda _ref: ("error", "status: error"),
        on_status=lambda *_args: None,
        fetch_logs=lambda _ref: "worker traceback",
        failure_factory=lambda logs: ValueError("FAILED:" + logs),
        timeout_message="timeout",
        now=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    raise AssertionError("terminal error accepted")
except ValueError as exc:
    assert "worker traceback" in str(exc)

try:
    wait_for_kernel(
        "user/kernel",
        kernel_timeout=60,
        poll_seconds=1,
        check_cancelled=lambda: None,
        kernel_status=lambda _ref: ("error", "raw fallback"),
        on_status=lambda *_args: None,
        fetch_logs=lambda _ref: (_ for _ in ()).throw(RuntimeError("logs unavailable")),
        failure_factory=lambda logs: ValueError(logs),
        timeout_message="timeout",
        now=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    raise AssertionError("terminal error fallback accepted")
except ValueError as exc:
    assert "raw fallback" in str(exc)

ticks = iter([0.0, 1261.0])
try:
    wait_for_kernel(
        "user/kernel",
        kernel_timeout=60,
        poll_seconds=1,
        check_cancelled=lambda: None,
        kernel_status=lambda _ref: ("running", "status: running"),
        on_status=lambda *_args: None,
        fetch_logs=lambda _ref: "",
        failure_factory=lambda logs: RuntimeError(logs),
        timeout_message="custom timeout",
        now=lambda: next(ticks),
        sleep=lambda _seconds: None,
    )
    raise AssertionError("timeout not enforced")
except TimeoutError as exc:
    assert "custom timeout" in str(exc)

print("Kaggle kernel monitor tests passed.")

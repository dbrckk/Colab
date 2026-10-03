from __future__ import annotations

from typing import Callable


KernelStatus = Callable[[str], tuple[str, str]]
CancelCheck = Callable[[], None]
StatusCallback = Callable[[str, str], None]
LogsFetcher = Callable[[str], str]
FailureFactory = Callable[[str], Exception]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]


def wait_for_kernel(
    kernel_ref: str,
    *,
    kernel_timeout: int,
    poll_seconds: int,
    check_cancelled: CancelCheck,
    kernel_status: KernelStatus,
    on_status: StatusCallback,
    fetch_logs: LogsFetcher,
    failure_factory: FailureFactory,
    timeout_message: str,
    now: Clock,
    sleep: Sleeper,
) -> str:
    started = now()

    while True:
        check_cancelled()
        if now() - started > kernel_timeout + 1200:
            raise TimeoutError(timeout_message)

        state, raw = kernel_status(kernel_ref)
        on_status(state, raw)

        if state == "complete":
            return raw

        if state == "error":
            try:
                logs = fetch_logs(kernel_ref)
            except Exception:
                logs = raw
            raise failure_factory(logs)

        sleep(max(5, poll_seconds))

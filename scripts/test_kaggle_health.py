from kaggle_app.kaggle_health import recovery_diagnostic_lines
from kaggle_app.kaggle_recovery import recovery_marked, remote_kernel_is_preserved


jobs = [
    {"status": "waiting_auth", "kernel_ref": "u/k1", "meta": {"recover_outputs_available": True}},
    {"status": "error", "kernel_ref": "u/k2", "meta": {"recover_outputs_available": True}},
    {"status": "done", "kernel_ref": "", "meta": {}},
]

lines = recovery_diagnostic_lines(
    jobs,
    recovery_marked=recovery_marked,
    remote_kernel_is_preserved=remote_kernel_is_preserved,
)
text = "\n".join(lines)
assert "2 job(s) ont des outputs Kaggle récupérables" in text
assert "1 job(s) attendent l'authentification" in text
assert "2 kernel(s) distant(s) conservé(s)" in text

empty = recovery_diagnostic_lines(
    [{"status": "done", "kernel_ref": "", "meta": {}}],
    recovery_marked=recovery_marked,
    remote_kernel_is_preserved=remote_kernel_is_preserved,
)
assert empty == ["✅ Aucun job de reprise en attente."]

print("Kaggle recovery health summary tests passed.")

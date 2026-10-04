from kaggle_app.kaggle_dashboard import (
    dashboard_summary_text,
    format_storage_bytes,
)


jobs = [
    {"status": "queued"},
    {"status": "running"},
    {"status": "waiting_auth"},
    {"status": "done"},
    {"status": "done"},
    {"status": "error"},
    {"status": "interrupted"},
    {"status": "cancelled"},
]

summary = dashboard_summary_text(
    jobs,
    total_bytes=512 * 1024**2,
    auth_label="Kaggle prêt",
)

assert "Kaggle prêt" in summary
assert "Actifs **3**" in summary
assert "Terminés **2**" in summary
assert "Erreurs **2**" in summary
assert "Annulés **1**" in summary
assert "Stockage **512.0 Mo**" in summary

assert format_storage_bytes(0) == "0.0 Mo"
assert format_storage_bytes(1024**3) == "1.00 Go"
assert format_storage_bytes(1536 * 1024**2) == "1.50 Go"

print("Kaggle dashboard summary tests passed.")

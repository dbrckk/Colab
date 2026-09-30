from kaggle_app.ui import build_ui

demo = build_ui()
assert demo is not None
print("Kaggle Studio UI smoke test passed.")

import tempfile
from pathlib import Path

from kaggle_app.kaggle_input_store import (
    ensure_input_store,
    hash_file,
    persist_input,
    unreferenced_store_files,
)


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    source_a = root / "a.png"
    source_b = root / "renamed.jpg"
    source_a.write_bytes(b"same-bytes")
    source_b.write_bytes(b"same-bytes")

    stored_a = Path(persist_input(root, str(source_a)))
    stored_b = Path(persist_input(root, str(source_b)))

    assert stored_a == stored_b
    assert stored_a.exists()
    assert stored_a.name.startswith(hash_file(source_a))
    assert persist_input(root, str(stored_a)) == str(stored_a)

    store = ensure_input_store(root)
    extra = store / "deadbeef.bin"
    extra.write_bytes(b"orphan")
    hidden = store / ".partial.part"
    hidden.write_bytes(b"partial")

    stale = unreferenced_store_files(
        root,
        referenced_paths={str(stored_a.resolve())},
    )
    assert [path.name for path in stale] == ["deadbeef.bin"]

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    missing = root / "missing.png"
    try:
        persist_input(root, str(missing))
        raise AssertionError("missing input accepted")
    except FileNotFoundError:
        pass

print("Kaggle content-addressed input store tests passed.")

import json

import pytest

from modelpicker.data import import_local, import_url, parse_file, sample_rows, save_dataset, summarize
from modelpicker.storage import Store


def test_ingestion_is_strict_and_preview_is_opt_in(settings, rows):
    store = Store(settings.data_dir)
    result = save_dataset(store, settings, rows, "test", "text", "label", {"type": "test"})
    assert result["row_count"] == 20
    assert result["labels"] == {"negative": 10, "positive": 10}
    assert "rows" not in result and "preview" not in result
    saved = store.get("dataset", result["id"])
    assert summarize(saved, True)["preview"][0]["text"] == "example 0"
    assert len(result["sha256"]) == 64
    with pytest.raises(ValueError, match="invalid rows"):
        save_dataset(store, settings, rows + [{"text": "missing label"}], "bad", "text", "label", {})
    assert len(store.list("dataset")) == 1  # Failed import is atomic.


def test_custom_column_csv_jsonl_and_authorized_paths(settings):
    store = Store(settings.data_dir)
    (settings.dataset_dir / "data.csv").write_text(
        'review,sentiment\n"hello, world",positive\nbad,negative\n'
    )
    result = import_local(store, settings, "data.csv", "review", "sentiment")
    assert result["row_count"] == 2
    assert parse_file('{"text":"great","label":1}\n', "jsonl")[0]["label"] == 1
    outside = settings.data_dir / "secret.json"
    outside.write_text(json.dumps([{"text": "secret", "label": "secret"}]))
    with pytest.raises(ValueError, match="inside"):
        import_local(store, settings, str(outside), "text", "label")
    (settings.dataset_dir / "escape.json").symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        import_local(store, settings, "escape.json", "text", "label")


def test_sampling_is_exact_repeatable_and_does_not_mutate(rows):
    normalized = [{"row_id": i, **r} for i, r in enumerate(rows)]
    original = list(normalized)
    for method in ("random", "representative", "balanced"):
        first = sample_rows(normalized, 9, method, 42)
        assert first == sample_rows(normalized, 9, method, 42)
        assert len(first) == len({r["row_id"] for r in first}) == 9
    assert normalized == original
    assert sample_rows(normalized, 20, "all", 42) == normalized
    with pytest.raises(ValueError):
        sample_rows(normalized, 21, "random", 42)
    with pytest.raises(ValueError):
        sample_rows(normalized, 5, "all", 42)
    unbalanced = [{"row_id": i, "label": "a" if i < 9 else "b"} for i in range(10)]
    with pytest.raises(ValueError, match="Not enough"):
        sample_rows(unbalanced, 8, "balanced", 42)
    picked = sample_rows(unbalanced, 8, "representative", 42)
    assert sum(r["label"] == "a" for r in picked) == 7


def test_url_import_rejects_private_and_unapproved_hosts(settings, monkeypatch):
    store = Store(settings.data_dir)
    for url in (
        "http://raw.githubusercontent.com/test.csv",
        "https://127.0.0.1/data.csv",
        "https://user:pass@raw.githubusercontent.com/test.csv",
        "https://evil.example/data.csv",
    ):
        with pytest.raises(ValueError, match="HTTPS"):
            import_url(store, settings, url, "csv", "text", "label")
    monkeypatch.setattr(
        "modelpicker.data.socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 443))]
    )
    with pytest.raises(ValueError, match="private"):
        import_url(store, settings, "https://raw.githubusercontent.com/a.csv", "csv", "text", "label")

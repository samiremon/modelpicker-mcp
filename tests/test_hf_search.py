from types import SimpleNamespace

from guards import hf_search


def fake_model(model_id, downloads, total=None):
    safetensors = SimpleNamespace(total=total) if total else None
    return SimpleNamespace(id=model_id, downloads=downloads, likes=1, safetensors=safetensors)


def use_models(monkeypatch, models):
    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_models(self, **kwargs):
            return iter(models)

    monkeypatch.setattr(hf_search, "HfApi", FakeApi)


def test_skips_rarely_downloaded_and_oversized_models(monkeypatch):
    use_models(
        monkeypatch,
        [
            fake_model("tiny/unused", 10, 1_000_000),
            fake_model("big/model", 5000, 900_000_000),
            fake_model("good/model", 5000, 184_400_000),
        ],
    )
    results = hf_search.search_models("prompt injection", max_params_million=300)
    assert [m["model_id"] for m in results] == ["good/model"]
    assert results[0]["params_million"] == 184.4


def test_unknown_size_is_reported_not_guessed(monkeypatch):
    use_models(monkeypatch, [fake_model("no/size", 5000)])
    assert hf_search.search_models("prompt injection")[0]["params_million"] is None


def test_stops_at_top_k(monkeypatch):
    use_models(monkeypatch, [fake_model(f"m/{i}", 5000) for i in range(5)])
    assert len(hf_search.search_models("prompt injection", top_k=2)) == 2

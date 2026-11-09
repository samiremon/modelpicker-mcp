import pytest

from modelpicker.config import Settings
from modelpicker.engine import Engine


@pytest.fixture
def settings(tmp_path):
    value = Settings(
        data_dir=tmp_path / "state",
        dataset_dir=tmp_path / "datasets",
        cache_dir=tmp_path / "models",
        cpu_threads=1,
    )
    value.prepare()
    return value


@pytest.fixture
def engine(settings):
    value = Engine(settings)
    yield value
    value.close()


@pytest.fixture
def rows():
    return [{"text": f"example {i}", "label": "positive" if i % 2 else "negative"} for i in range(20)]


@pytest.fixture
def fake_inspect(monkeypatch):
    def inspect(model_id, settings):
        return {
            "model_id": model_id,
            "revision": "a" * 40,
            "weight_size_mb": 1,
            "output_labels": {"0": "NEGATIVE", "1": "POSITIVE"},
        }

    monkeypatch.setattr("modelpicker.engine.inspect_model", inspect)
    return inspect

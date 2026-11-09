import pytest

from modelpicker.metrics import classification_metrics, recommend
from modelpicker.models import resolve_mapping, validate_model_id


def test_metrics_include_failed_predictions_and_macro_f1():
    predictions = [
        {"expected": "a", "predicted": "a", "latency_ms": 10},
        {"expected": "a", "predicted": "b", "latency_ms": 20},
        {"expected": "b", "predicted": "b", "latency_ms": 30},
        {"expected": "b", "predicted": None, "error": "timeout", "latency_ms": 1000},
    ]
    m = classification_metrics(predictions, ["a", "b"])
    assert m["accuracy"] == 0.5
    assert m["error_rate"] == 0.25
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert m["p95_latency_ms"] == pytest.approx(29)
    assert m["confusion_matrix"]["b"]["__ERROR__"] == 1
    assert m["accuracy_wilson_95"][0] < 0.5 < m["accuracy_wilson_95"][1]


def test_recommendation_enforces_all_constraints():
    def result(name, f1, latency, errors=0):
        return {
            "model_id": name,
            "revision": "abc",
            "status": "completed",
            "metrics": {"macro_f1": f1, "p95_latency_ms": latency, "error_count": errors},
        }

    results = [result("fast", 0.8, 10), result("quality", 0.95, 30), result("broken", 1, 1, 1)]
    assert recommend(results, 0.7, 50, "quality")["model_id"] == "quality"
    assert recommend(results, 0.7, 50, "speed")["model_id"] == "fast"
    assert recommend(results, 0.9, 20, "quality")["model_id"] is None


def test_label_mapping_never_guesses_polarity():
    generic = {"0": "LABEL_0", "1": "LABEL_1"}
    with pytest.raises(ValueError, match="explicit"):
        resolve_mapping(generic, ["negative", "positive"], None)
    assert (
        resolve_mapping(generic, ["negative", "positive"], {"LABEL_0": "negative", "LABEL_1": "positive"})[
            "LABEL_0"
        ]
        == "negative"
    )
    with pytest.raises(ValueError):
        resolve_mapping(generic, ["negative", "positive"], {"LABEL_0": "positive", "LABEL_1": "positive"})
    with pytest.raises(ValueError):
        resolve_mapping({"0": "negative", "1": "neutral", "2": "positive"}, ["negative", "positive"], None)
    for invalid in ("../secret", "/local/path", "https://example.com/model"):
        with pytest.raises(ValueError):
            validate_model_id(invalid)

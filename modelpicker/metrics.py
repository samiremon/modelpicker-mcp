from __future__ import annotations

import math
from collections import Counter


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def classification_metrics(predictions: list[dict], labels: list[str]) -> dict:
    count = len(predictions)
    correct = sum(p.get("predicted") == p["expected"] and not p.get("error") for p in predictions)
    errors = sum(bool(p.get("error")) for p in predictions)
    per_label = {}
    confusion = {
        label: dict(Counter(p.get("predicted") or "__ERROR__" for p in predictions if p["expected"] == label))
        for label in labels
    }
    for name in labels:
        tp = sum(p.get("predicted") == name and p["expected"] == name for p in predictions)
        fp = sum(p.get("predicted") == name and p["expected"] != name for p in predictions)
        fn = sum(p["expected"] == name and p.get("predicted") != name for p in predictions)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[name] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    accuracy = correct / count if count else 0.0
    # Wilson interval for accuracy; no independence claim for duplicate/related examples.
    if count:
        z = 1.96
        center = (accuracy + z * z / (2 * count)) / (1 + z * z / count)
        half = (
            z
            * math.sqrt(accuracy * (1 - accuracy) / count + z * z / (4 * count * count))
            / (1 + z * z / count)
        )
        interval = [max(0, center - half), min(1, center + half)]
    else:
        interval = None
    latencies = [p["latency_ms"] for p in predictions if not p.get("error")]
    return {
        "rows": count,
        "accuracy": accuracy,
        "accuracy_wilson_95": interval,
        "macro_f1": sum(v["f1"] for v in per_label.values()) / len(labels),
        "error_count": errors,
        "error_rate": errors / count if count else 0,
        "p50_latency_ms": percentile(latencies, 0.5),
        "p95_latency_ms": percentile(latencies, 0.95),
        "successful_predictions_per_second": len(latencies) / (sum(latencies) / 1000)
        if sum(latencies)
        else 0,
        "per_label": per_label,
        "confusion_matrix": confusion,
        "latency_scope": "Sequential, batch size 1, tokenization + CPU inference; excludes warmup/download/loading",
    }


def recommend(results: list[dict], min_macro_f1: float, max_p95_ms: float | None, priority: str) -> dict:
    passing = [
        r
        for r in results
        if r.get("status") == "completed"
        and r["metrics"]["error_count"] == 0
        and r["metrics"]["macro_f1"] >= min_macro_f1
        and (
            max_p95_ms is None
            or (r["metrics"]["p95_latency_ms"] is not None and r["metrics"]["p95_latency_ms"] <= max_p95_ms)
        )
    ]
    if not passing:
        return {
            "model_id": None,
            "reason": "No fully evaluated model passed all quality, latency, and error requirements.",
        }
    if priority == "speed":
        passing.sort(key=lambda r: (r["metrics"]["p95_latency_ms"], -r["metrics"]["macro_f1"]))
    else:
        passing.sort(key=lambda r: (-r["metrics"]["macro_f1"], r["metrics"]["p95_latency_ms"]))
    return {
        "model_id": passing[0]["model_id"],
        "revision": passing[0]["revision"],
        "reason": f"Passed all requirements; ranked by {priority} on this sample and machine.",
        "caution": "Exploratory selection, not held-out proof. Validate on fresh representative examples before deployment.",
    }

"""Isolated CPU inference process. Invoked by the job manager, never by an MCP client."""

from __future__ import annotations

import gc
import json
import platform
import sys
import time
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def evaluate(request: dict, progress) -> dict:
    import psutil
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    from .config import hf_token
    from .metrics import classification_metrics, recommend

    settings, plan, rows = request["settings"], request["plan"], request["rows"]
    torch.set_num_threads(settings["cpu_threads"])
    torch.manual_seed(plan["seed"])
    results = []
    completed = 0
    elapsed_predictions, measured_predictions = 0.0, 0
    started = time.perf_counter()
    for spec in plan["models"]:
        result = {"model_id": spec["model_id"], "revision": spec["revision"], "status": "failed"}
        classifier = model = tokenizer = None
        try:
            progress(
                {
                    "phase": "loading",
                    "model_id": spec["model_id"],
                    "completed_predictions": completed,
                    "remaining_seconds_estimate": None,
                }
            )
            if psutil.virtual_memory().available / 1024**2 < settings["min_available_ram_mb"]:
                raise ValueError("Not enough available RAM; close applications or choose a smaller model")
            load_start = time.perf_counter()
            kwargs = {
                "revision": spec["revision"],
                "cache_dir": settings["cache_dir"],
                "trust_remote_code": False,
                "token": hf_token(),
            }
            tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], **kwargs)
            model, loading = AutoModelForSequenceClassification.from_pretrained(
                spec["model_id"], use_safetensors=True, output_loading_info=True, **kwargs
            )
            if loading.get("missing_keys") or loading.get("mismatched_keys"):
                raise ValueError(
                    "Checkpoint has missing/mismatched weights; refusing an untrained classification head"
                )
            classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
            result["load_seconds"] = time.perf_counter() - load_start
            raw_limit = getattr(tokenizer, "model_max_length", 512)
            max_length = min(plan["max_length"], raw_limit if raw_limit < 100000 else 512)
            result["effective_max_length"] = max_length
            # Warmup is measured separately and excluded from per-example latency.
            warmup_start = time.perf_counter()
            classifier(rows[0]["text"], truncation=True, max_length=max_length)
            result["warmup_seconds"] = time.perf_counter() - warmup_start
            predictions = []
            max_rss = psutil.Process().memory_info().rss
            for row in rows:
                before = time.perf_counter()
                prediction = {"row_id": row["row_id"], "expected": row["label"]}
                try:
                    output = classifier(row["text"], truncation=True, max_length=max_length)[0]
                    raw = output["label"]
                    mapped = spec["label_mapping"].get(raw)
                    if mapped is None:
                        raise ValueError(f"Unexpected output label: {raw}")
                    prediction.update(predicted=mapped, raw_label=raw, confidence=float(output["score"]))
                except Exception as exc:
                    prediction.update(predicted=None, error=f"{type(exc).__name__}: {str(exc)[:300]}")
                prediction["latency_ms"] = (time.perf_counter() - before) * 1000
                predictions.append(prediction)
                elapsed_predictions += prediction["latency_ms"] / 1000
                measured_predictions += 1
                completed += 1
                max_rss = max(max_rss, psutil.Process().memory_info().rss)
                remaining = plan["total_predictions"] - completed
                progress(
                    {
                        "phase": "evaluating",
                        "model_id": spec["model_id"],
                        "completed_predictions": completed,
                        "remaining_seconds_estimate": round(
                            remaining * elapsed_predictions / measured_predictions, 1
                        ),
                        "estimate_note": "Based on observed rows; excludes loading remaining models.",
                    }
                )
            result.update(
                status="completed",
                predictions=predictions,
                metrics=classification_metrics(predictions, plan["labels"]),
                observed_process_rss_mb=round(max_rss / 1024**2, 1),
                actual_params=sum(p.numel() for p in model.parameters()),
                weight_size_mb=spec["weight_size_mb"],
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {str(exc)[:700]}"
        finally:
            classifier = model = tokenizer = None
            gc.collect()
        results.append(result)
    return {
        "results": results,
        "recommendation": recommend(results, plan["min_macro_f1"], plan["max_p95_ms"], plan["priority"]),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "cpu_threads": settings["cpu_threads"],
            "batch_size": 1,
        },
        "notes": [
            "RSS is the largest process RSS observed after each example; not isolated model memory or a continuous peak.",
            "Inputs are truncated to each model's effective_max_length; latency comparisons reflect this setting.",
            "No API inference charges. Electricity, internet, and hardware are supplied by the user.",
            "No accuracy claim beyond the selected dataset. Model training contamination is not checked.",
        ],
    }


def main():
    directory = Path(sys.argv[1])
    try:
        request = json.loads((directory / "input.json").read_text(encoding="utf-8"))
        result = evaluate(request, lambda value: atomic_json(directory / "progress.json", value))
        atomic_json(directory / "result.json", result)
    except Exception as exc:
        atomic_json(directory / "result.json", {"fatal_error": f"{type(exc).__name__}: {str(exc)[:700]}"})


if __name__ == "__main__":
    main()

from __future__ import annotations

import json

from .storage import Store


def cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("`", "'")


def export_report(store: Store, job_id: str, format: str = "markdown") -> dict:
    job = store.get("job", job_id)
    plan = store.get("plan", job["plan_id"])
    if "output" not in job:
        raise ValueError("Evaluation has no completed results to export")
    # No raw dataset texts or predictions in a default report.
    output = {
        **job["output"],
        "results": [{k: v for k, v in r.items() if k != "predictions"} for r in job["output"]["results"]],
    }
    if format == "json":
        content = json.dumps(
            {"job_id": job_id, "status": job["status"], "plan": plan, "output": output}, indent=2
        )
        return {"filename": f"modelpicker-{job_id}.json", "mime_type": "application/json", "content": content}
    if format != "markdown":
        raise ValueError("format must be markdown or json")
    lines = [
        "# ModelPicker measured evaluation",
        "",
        f"Dataset: {cell(plan['dataset_name'])} · Rows: {plan['sample_size']} · Status: {job['status']}",
        f"Dataset SHA256: `{plan['dataset_sha256']}`",
        f"Sample SHA256: `{plan['sample_sha256']}`",
        "",
        "| Model | Status | Macro F1 | Accuracy | p50 ms | p95 ms | Errors |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in output["results"]:
        m = result.get("metrics")
        if not m:
            lines.append(f"| {cell(result['model_id'])} | failed | — | — | — | — | — |")
            continue
        p50 = f"{m['p50_latency_ms']:.2f}" if m["p50_latency_ms"] is not None else "—"
        p95 = f"{m['p95_latency_ms']:.2f}" if m["p95_latency_ms"] is not None else "—"
        lines.append(
            f"| {cell(result['model_id'])} | {result['status']} | {m['macro_f1']:.4f} | "
            f"{m['accuracy']:.4f} | {p50} | {p95} | {m['error_count']} |"
        )
    rec = output["recommendation"]
    lines += [
        "",
        f"Recommendation: {cell(rec.get('model_id') or 'None')}. {rec['reason']}",
        "",
        "Timing: CPU, sequential requests, batch size 1; excludes downloads, loading, and warmup.",
        "RAM: observed whole-process RSS, not isolated model memory.",
        "Quality: exploratory sample results; validate on fresh held-out examples.",
        "",
        "## Reproduction",
        "",
        "```json",
        json.dumps({"plan": plan, "environment": output["environment"]}, indent=2),
        "```",
    ]
    for result in output["results"]:
        if result.get("error"):
            lines += ["", f"{cell(result['model_id'])}: {cell(result['error'])}"]
    return {"filename": f"modelpicker-{job_id}.md", "mime_type": "text/markdown", "content": "\n".join(lines)}


def export_integration(store: Store, job_id: str, model_id: str | None = None) -> dict:
    job = store.get("job", job_id)
    output = job.get("output", {})
    model_id = model_id or output.get("recommendation", {}).get("model_id")
    if not model_id:
        raise ValueError("No recommended model; inspect results or specify a successfully evaluated model")
    result = next(
        (r for r in output.get("results", []) if r["model_id"] == model_id and r["status"] == "completed"),
        None,
    )
    if result is None:
        raise ValueError("Choose a successfully evaluated model")
    plan = store.get("plan", job["plan_id"])
    spec = next(m for m in plan["models"] if m["model_id"] == model_id)
    # repr escapes repository names/labels as literals; never interpolate executable user code.
    code = f'''"""Generated from ModelPicker run {job_id}. Install torch and transformers."""
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_ID = {model_id!r}
REVISION = {spec["revision"]!r}
LABEL_MAPPING = {spec["label_mapping"]!r}
torch.set_num_threads({output["environment"]["cpu_threads"]})
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False,
                                         cache_dir=os.getenv("MODELPICKER_CACHE_DIR"))
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, revision=REVISION, trust_remote_code=False, use_safetensors=True,
    cache_dir=os.getenv("MODELPICKER_CACHE_DIR"))
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)

def classify(text: str) -> dict:
    result = classifier(text, truncation=True, max_length={result["effective_max_length"]})[0]
    return {{"label": LABEL_MAPPING[result["label"]], "confidence": result["score"]}}

if __name__ == "__main__":
    print(classify("Replace this with your input text."))
'''
    return {
        "filename": "modelpicker_integration.py",
        "content": code,
        "note": "Uses the tested revision, mapping, and truncation. Validate on fresh examples before deployment.",
    }


def compare_runs(store: Store, first_id: str, second_id: str) -> dict:
    first, second = (store.get("job", i) for i in (first_id, second_id))
    a, b = (store.get("plan", j["plan_id"]) for j in (first, second))
    fair = a["sample_sha256"] == b["sample_sha256"] and a["max_length"] == b["max_length"]
    ar = {r["model_id"]: r for r in first.get("output", {}).get("results", []) if r.get("metrics")}
    br = {r["model_id"]: r for r in second.get("output", {}).get("results", []) if r.get("metrics")}
    deltas = []
    for model in sorted(ar.keys() & br.keys()):
        am, bm = ar[model]["metrics"], br[model]["metrics"]
        deltas.append(
            {
                "model_id": model,
                "macro_f1_change": bm["macro_f1"] - am["macro_f1"],
                "accuracy_change": bm["accuracy"] - am["accuracy"],
                "p95_latency_ms_change": bm["p95_latency_ms"] - am["p95_latency_ms"]
                if bm["p95_latency_ms"] is not None and am["p95_latency_ms"] is not None
                else None,
                "same_revision": ar[model]["revision"] == br[model]["revision"],
            }
        )
    return {
        "same_sample_and_token_limit": fair,
        "changes_second_minus_first": deltas,
        "same_environment": first.get("output", {}).get("environment")
        == second.get("output", {}).get("environment"),
        "note": "Timing varies with machine load. Different samples/settings are not a controlled comparison.",
    }

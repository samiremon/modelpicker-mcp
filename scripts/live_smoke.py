"""Live CPU validation, with real Hub downloads. No paid APIs.

Run from the installed environment: python scripts/live_smoke.py [--hub] [--model MODEL_ID ...]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from modelpicker.config import Settings
from modelpicker.data import import_hf, import_local
from modelpicker.engine import TERMINAL, Engine
from modelpicker.reports import export_integration, export_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hub", action="store_true", help="Import 100 SST-2 validation rows instead of authored demo.csv"
    )
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = Settings(
        data_dir=root / ".demo-data", dataset_dir=root / "examples", cache_dir=root / ".demo-data" / "models"
    )
    engine = Engine(settings)
    try:
        if args.hub:
            dataset = import_hf(
                engine.store, settings, "stanfordnlp/sst2", None, "validation", "sentence", "label", 100, 42
            )
        else:
            dataset = import_local(engine.store, settings, "demo.csv", "text", "label")
        models = args.models or ["distilbert/distilbert-base-uncased-finetuned-sst-2-english"]
        plan = engine.create_plan(dataset["id"], models, sample_size=dataset["row_count"], sampling="all")
        job = engine.start(plan["id"])
        print(f"Job {job['id']}: {plan['total_predictions']} real predictions", flush=True)
        last = None
        while True:
            job = engine.get_job(job["id"])
            snapshot = (
                job["status"],
                job.get("progress", {}).get("phase"),
                job.get("progress", {}).get("completed_predictions"),
            )
            if snapshot != last:
                print(snapshot, flush=True)
                last = snapshot
            if job["status"] in TERMINAL:
                break
            time.sleep(1)
        if job["status"] != "completed":
            raise RuntimeError(json.dumps(job, indent=2))
        for result in job["output"]["results"]:
            assert result["metrics"]["rows"] == dataset["row_count"]
            assert result["metrics"]["error_count"] == 0
            assert result["metrics"]["p95_latency_ms"] > 0
        report = export_report(engine.store, job["id"])
        (settings.data_dir / "last-report.md").write_text(report["content"], encoding="utf-8")
        integration = export_integration(engine.store, job["id"])
        compile(integration["content"], "modelpicker_integration.py", "exec")
        # Execute generated code against the same cache to verify it really loads and predicts.
        import os

        os.environ["MODELPICKER_CACHE_DIR"] = str(settings.cache_dir)
        namespace = {"__name__": "integration_smoke"}
        exec(integration["content"], namespace)
        prediction = namespace["classify"]("The delivery was excellent and I am very happy.")
        assert prediction["label"] in plan["labels"]
        print(json.dumps(job["output"], indent=2), flush=True)
        print(
            f"Live evaluation and generated integration passed. Report: {settings.data_dir / 'last-report.md'}"
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()

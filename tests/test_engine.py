import json
import time

import pytest

from modelpicker.data import save_dataset
from modelpicker.engine import Engine
from modelpicker.reports import compare_runs, export_integration, export_report


def plan_for(engine, rows):
    dataset = save_dataset(engine.store, engine.settings, rows, "dataset", "text", "label", {"type": "test"})
    plan = engine.create_plan(dataset["id"], ["owner/model", "owner/model2"], sample_size=10)
    return dataset, plan


def test_plan_has_shared_rows_pins_and_no_fabricated_timing(engine, rows, fake_inspect):
    dataset, plan = plan_for(engine, rows)
    assert plan["total_predictions"] == 20
    assert len(set(plan["row_ids"])) == 10
    assert plan["models"][0]["revision"] == "a" * 40
    assert plan["estimated_seconds"] is None
    assert plan["api_inference_cost_usd"] == 0
    with pytest.raises(ValueError, match="referencing plans"):
        engine.delete("dataset", dataset["id"])
    schedule = engine.schedule(plan["id"])
    assert schedule["enabled"]
    job = engine.store.create("job", {"plan_id": plan["id"], "status": "completed"})
    engine.store.update("schedule", schedule["id"], last_job_id=job["id"])
    engine.delete("job", job["id"])
    assert engine.store.get("schedule", schedule["id"])["last_job_id"] is None
    engine.delete("schedule", schedule["id"])
    engine.delete("plan", plan["id"])
    engine.delete("dataset", dataset["id"])


def test_double_service_rejected_and_restart_marks_interrupted(settings):
    first = Engine(settings)
    try:
        first.store.create("job", {"plan_id": "missing", "status": "running"})
        with pytest.raises(ValueError, match="Another"):
            Engine(settings)
    finally:
        first.close()
    second = Engine(settings)
    try:
        assert second.store.list("job")[0]["status"] == "interrupted"
    finally:
        second.close()


def test_background_worker_persistence_cancellation_and_timeout(engine, rows, fake_inspect, monkeypatch):
    _, plan = plan_for(engine, rows)

    # Exercise process supervision without downloading models or faking evaluation quality.
    class SleepingProcess:
        pid = 999999999
        returncode = None

        def __init__(self, *a, **kw):
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode

    monkeypatch.setattr("modelpicker.engine.subprocess.Popen", SleepingProcess)
    monkeypatch.setattr("modelpicker.engine.importlib.util.find_spec", lambda _: True)
    monkeypatch.setattr(engine, "_terminate", lambda p: p.terminate())
    job = engine.start(plan["id"])
    deadline = time.time() + 5
    while engine.get_job(job["id"])["status"] == "queued" and time.time() < deadline:
        time.sleep(0.02)
    assert engine.get_job(job["id"])["status"] == "running"
    engine.cancel(job["id"])
    while engine.get_job(job["id"])["status"] not in {"cancelled", "failed"} and time.time() < deadline:
        time.sleep(0.02)
    assert engine.get_job(job["id"])["status"] == "cancelled"
    assert not (engine.settings.data_dir / "work" / job["id"]).exists()


def test_export_failures_and_comparison_do_not_leak_raw_text(engine, rows, fake_inspect):
    _, plan = plan_for(engine, rows)
    result = {
        "model_id": "owner/model",
        "revision": "a" * 40,
        "status": "completed",
        "effective_max_length": 256,
        "metrics": {
            "accuracy": 0.5,
            "macro_f1": 0.5,
            "p50_latency_ms": 10,
            "p95_latency_ms": 20,
            "error_count": 0,
        },
        "predictions": [{"row_id": 0, "expected": "negative", "predicted": "positive", "latency_ms": 10}],
    }
    output = {
        "results": [result],
        "recommendation": {"model_id": "owner/model", "reason": "test"},
        "environment": {"cpu_threads": 1},
    }
    job = engine.store.create("job", {"plan_id": plan["id"], "status": "completed", "output": output})
    assert "predictions" not in engine.get_job(job["id"])["output"]["results"][0]
    assert "text" not in engine.failures(job["id"], "owner/model")["failures"][0]
    assert engine.failures(job["id"], "owner/model", include_text=True)["failures"][0]["text"] == "example 0"
    report = export_report(engine.store, job["id"], "json")
    assert "predictions" not in json.loads(report["content"])["output"]["results"][0]
    assert json.loads(report["content"])["plan"]["sample_size"] == 10
    assert "measured" in export_report(engine.store, job["id"])["content"]
    compile(export_integration(engine.store, job["id"])["content"], "integration.py", "exec")
    assert compare_runs(engine.store, job["id"], job["id"])["same_sample_and_token_limit"]

import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelpicker.data import save_dataset
from modelpicker.engine import TERMINAL


class Process:
    pid = 999999998
    returncode = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15


def wait_for_job(engine, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = engine.get_job(job_id)
        if job["status"] in TERMINAL:
            return job
        time.sleep(0.02)
    raise AssertionError("Job did not finish")


def make_plan(engine, rows):
    dataset = save_dataset(engine.store, engine.settings, rows, "supervision", "text", "label", {})
    return engine.create_plan(dataset["id"], ["owner/a", "owner/b"], sample_size=10)


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["completed", "completed"], "completed"),
        (["completed", "failed"], "partial"),
        (["failed", "failed"], "failed"),
    ],
)
def test_result_aggregation(engine, rows, fake_inspect, monkeypatch, statuses, expected):
    plan = make_plan(engine, rows)
    monkeypatch.setattr("modelpicker.engine.importlib.util.find_spec", lambda _: True)

    def launch(command, **kwargs):
        directory = Path(command[-1])
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "results": [
                        {"model_id": name, "status": status}
                        for name, status in zip(["owner/a", "owner/b"], statuses)
                    ],
                    "recommendation": {"model_id": None},
                }
            )
        )
        return Process()

    monkeypatch.setattr("modelpicker.engine.subprocess.Popen", launch)
    job = wait_for_job(engine, engine.start(plan["id"])["id"])
    assert job["status"] == expected


@pytest.mark.parametrize("reason", ["timeout", "memory", "crash"])
def test_supervisor_bounds_failures(engine, rows, fake_inspect, monkeypatch, reason):
    plan = make_plan(engine, rows)
    monkeypatch.setattr("modelpicker.engine.importlib.util.find_spec", lambda _: True)
    process = Process()
    process.returncode = 9 if reason == "crash" else None
    monkeypatch.setattr("modelpicker.engine.subprocess.Popen", lambda *a, **kw: process)
    monkeypatch.setattr(engine, "_terminate", lambda p: p.terminate())
    if reason == "timeout":
        engine.settings = replace(engine.settings, max_job_seconds=1)
    if reason == "memory":
        monkeypatch.setattr(
            "modelpicker.engine.psutil.Process",
            lambda _: SimpleNamespace(
                create_time=lambda: 42, memory_info=lambda: SimpleNamespace(rss=8192 * 1024**2)
            ),
        )
    job = wait_for_job(engine, engine.start(plan["id"])["id"])
    assert job["status"] == "failed"
    assert {"timeout": "time limit", "memory": "Memory resource", "crash": "exited with code 9"}[
        reason
    ] in job["error"]

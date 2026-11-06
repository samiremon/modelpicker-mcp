"""Run attacks against an agent command and report what happened."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agents.connection import DEFAULT_MODEL
from scanner.attacks import all_attack_ids, load_attack
from scanner.report import format_report
from scanner.scorer import Outcome, Score, score

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
TIMEOUT_SECONDS = 600
MAX_CONSECUTIVE_ERRORS = 3  # Usually means the provider's quota has run out.


@dataclass
class RunResult:
    attack_id: str
    score: Score
    answer: str
    stderr: str


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")


def run_attack(attack_id: str, agent_command: str, timeout: float = TIMEOUT_SECONDS) -> RunResult:
    attack = load_attack(attack_id)
    with tempfile.TemporaryDirectory(prefix="agent-scan-") as run_dir:
        attack_file = Path(run_dir) / "attack.json"
        call_log = Path(run_dir) / "calls.jsonl"
        attack_file.write_text(json.dumps(attack), encoding="utf-8")
        env = {
            **os.environ,
            "AGENT_TASK": attack["task"],
            "BAIT_ATTACK_FILE": str(attack_file),
            "BAIT_CALL_LOG": str(call_log),
        }
        try:
            process = subprocess.run(
                shlex.split(agent_command),
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            exit_code, answer, stderr = process.returncode, process.stdout, process.stderr
        except subprocess.TimeoutExpired:
            exit_code, answer, stderr = None, "", f"Timed out after {timeout} seconds."
        calls = read_jsonl(call_log)
    return RunResult(attack_id, score(attack, calls, exit_code, answer), answer.strip(), stderr.strip())


def run_scan(scan_dir: Path, plan: dict, say=print) -> dict:
    """Run whatever the plan still needs. Results are saved as they finish, so a scan can resume."""
    results_path = scan_dir / "results.jsonl"
    done = Counter(result["attack_id"] for result in read_jsonl(results_path))
    pending = [
        attack_id
        for attack_id in plan["attacks"]
        for _ in range(max(0, plan["repeat"] - done[attack_id]))
    ]

    errors = consecutive_errors = 0
    stopped_early = False
    for position, attack_id in enumerate(pending, start=1):
        result = run_attack(attack_id, plan["agent"])
        say(f"[{position}/{len(pending)}] {attack_id}: {result.score.outcome.value}")
        if result.score.outcome is Outcome.ERROR:
            errors += 1
            consecutive_errors += 1
            say(f"  {result.stderr[-300:]}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                say(f"Stopping after {MAX_CONSECUTIVE_ERRORS} errors in a row.")
                stopped_early = True
                break
            continue
        consecutive_errors = 0
        append_jsonl(
            results_path,
            {
                "attack_id": attack_id,
                "outcome": result.score.outcome.value,
                "evidence": result.score.evidence,
                "answer": result.answer,
            },
        )
    return {"errors": errors, "stopped_early": stopped_early, "results": read_jsonl(results_path)}


def new_scan(args: argparse.Namespace) -> tuple[Path, dict]:
    attacks = [args.attack] if args.attack else all_attack_ids()
    scan_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    plan = {
        "attacks": attacks,
        "repeat": args.repeat,
        "agent": args.agent,
        "model": os.environ.get("AGENT_MODEL", DEFAULT_MODEL) if args.agent is DEFAULT_AGENT else None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    scan_dir = RUNS_DIR / scan_id
    scan_dir.mkdir(parents=True)
    (scan_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return scan_dir, plan


DEFAULT_AGENT = f"{shlex.quote(sys.executable)} -m agents.test_agent"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", help="Run one attack ID from attacks/. By default, runs them all.")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="Command that starts the agent")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per attack")
    parser.add_argument("--resume", metavar="SCAN_ID", help="Continue an earlier scan from runs/")
    args = parser.parse_args()

    if args.resume:
        scan_dir = RUNS_DIR / args.resume
        plan = json.loads((scan_dir / "plan.json").read_text(encoding="utf-8"))
    else:
        scan_dir, plan = new_scan(args)
    print(f"Scan {scan_dir.name}: {len(plan['attacks'])} attacks, {plan['repeat']} runs each")

    scan = run_scan(scan_dir, plan)
    print()
    if scan["results"]:
        print(format_report(scan["results"], scan["errors"]))

    finished = len(scan["results"]) == len(plan["attacks"]) * plan["repeat"]
    if not finished:
        print(f"\nIncomplete. To continue: python -m scanner.run --resume {scan_dir.name}")
    if any(r["outcome"] == Outcome.ATTACK_SUCCEEDED.value for r in scan["results"]):
        return 1
    return 0 if finished else 2


if __name__ == "__main__":
    sys.exit(main())

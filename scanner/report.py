"""Summarize scan results."""

import json

from scanner.scorer import Outcome

COLUMNS = [
    (Outcome.ATTACK_SUCCEEDED, "Succeeded"),
    (Outcome.RESISTED, "Resisted"),
    (Outcome.AGENT_STOPPED, "Stopped"),
    (Outcome.NOT_EXPOSED, "Not exposed"),
]


def success_rate(results: list[dict]) -> tuple[int, int]:
    """Only runs where the agent actually saw the attack can show whether it worked."""
    exposed = [r for r in results if r["outcome"] != Outcome.NOT_EXPOSED.value]
    return sum(r["outcome"] == Outcome.ATTACK_SUCCEEDED.value for r in exposed), len(exposed)


def guard_summary(results: list[dict], checks: list[dict]) -> list[str]:
    """A guard only gets credit for blocking the attacker's text. Other blocks are false alarms."""
    payload_runs = [r for r in results if any(c["payload"] for c in r["guard_checks"])]
    stopped = sum(any(c["payload"] and c["blocked"] for c in r["guard_checks"]) for r in payload_runs)
    harmless = [c for c in checks if not c["payload"]]
    false_alarms = sum(c["blocked"] for c in harmless)
    average = sum(c["ms"] for c in checks) / len(checks)
    return [
        f"Guard blocked the attack text: {stopped}/{len(payload_runs)} runs",
        f"Guard false alarms: {false_alarms}/{len(harmless)} harmless texts blocked",
        f"Guard speed: {average:.1f} ms per check",
    ]


def format_report(results: list[dict], error_count: int = 0) -> str:
    attack_ids = sorted({r["attack_id"] for r in results})
    width = max([len(a) for a in attack_ids] + [len("Attack")])
    headers = "  ".join(label.rjust(len(label)) for _, label in COLUMNS)
    lines = [f"{'Attack'.ljust(width)}  {headers}"]
    for attack_id in attack_ids:
        runs = [r for r in results if r["attack_id"] == attack_id]
        cells = "  ".join(
            str(sum(r["outcome"] == outcome.value for r in runs)).rjust(len(label))
            for outcome, label in COLUMNS
        )
        lines.append(f"{attack_id.ljust(width)}  {cells}")

    succeeded, exposed = success_rate(results)
    rate = f"{succeeded}/{exposed} runs ({succeeded / exposed:.0%})" if exposed else "no exposed runs"
    lines += ["", f"Attack success rate: {rate}"]
    if error_count:
        lines.append(f"Errors, excluded from the rate and retried on resume: {error_count}")

    checks = [check for r in results for check in r.get("guard_checks", [])]
    if checks:
        lines += ["", *guard_summary(results, checks)]

    evidence = [r for r in results if r["outcome"] == Outcome.ATTACK_SUCCEEDED.value]
    if evidence:
        lines += ["", "Evidence:"]
        lines += [f"  {r['attack_id']}: {json.dumps(r['evidence'])}" for r in evidence]
    return "\n".join(lines)

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

    evidence = [r for r in results if r["outcome"] == Outcome.ATTACK_SUCCEEDED.value]
    if evidence:
        lines += ["", "Evidence:"]
        lines += [f"  {r['attack_id']}: {json.dumps(r['evidence'])}" for r in evidence]
    return "\n".join(lines)

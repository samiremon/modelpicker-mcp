"""Check a guard against every attack and the harmless texts, without running an agent.

    uv run python -m guards.eval --guard keyword
"""

import argparse
from pathlib import Path

import yaml

from bait_server.server import tool_text
from guards import load_guard
from scanner.attacks import ATTACKS_DIR, all_attack_ids, load_attack

BENIGN_FILE = Path(__file__).resolve().parent / "benign.yaml"


def attack_texts(attack_id: str) -> list[str]:
    """Every text the attack itself adds. Shared tools are harmless and tested as benign."""
    shared = set(yaml.safe_load((ATTACKS_DIR / "_tools.yaml").read_text(encoding="utf-8")))
    texts = []
    for tool in load_attack(attack_id)["tools"]:
        if tool["name"] in shared:
            continue
        parameters = tool.get("parameters", {})
        texts.append(tool_text(tool["description"], parameters))
        if tool.get("description_after_first_call"):
            texts.append(tool_text(tool["description_after_first_call"], parameters))
        texts += tool["outputs"]
    return texts


def benign_texts() -> list[str]:
    return [item["text"] for item in yaml.safe_load(BENIGN_FILE.read_text(encoding="utf-8"))]


def evaluate(guard) -> dict:
    """An attack counts as caught if the guard flags any of its texts."""
    timings = []

    def flagged(text):
        verdict = guard.check(text)
        timings.append(verdict.ms)
        return verdict.is_attack

    caught = {a: any([flagged(t) for t in attack_texts(a)]) for a in all_attack_ids()}
    false_alarms = [text for text in benign_texts() if flagged(text)]
    return {
        "caught": caught,
        "false_alarms": false_alarms,
        "benign_total": len(benign_texts()),
        "ms_per_check": sum(timings) / len(timings),
    }


def format_eval(name: str, result: dict) -> str:
    caught = result["caught"]
    width = max(len(a) for a in caught)
    lines = [f"Guard: {name}", ""]
    lines += [f"  {a.ljust(width)}  {'caught' if hit else 'missed'}" for a, hit in caught.items()]
    lines += [
        "",
        f"Attacks caught: {sum(caught.values())}/{len(caught)}",
        f"False alarms:   {len(result['false_alarms'])}/{result['benign_total']} harmless texts",
        f"Speed:          {result['ms_per_check']:.1f} ms per check",
    ]
    lines += [f"  False alarm: {text.strip().splitlines()[0][:70]}" for text in result["false_alarms"]]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--guard", required=True, help="'keyword' or a Hugging Face model ID")
    args = parser.parse_args()
    print(format_eval(args.guard, evaluate(load_guard(args.guard))))


if __name__ == "__main__":
    main()

"""Check a guard against every attack and the harmless texts, without running an agent.

    uv run python -m guards.eval --guard keyword
"""

import argparse
from pathlib import Path

import yaml

from bait_server.server import tool_text
from guards import load_guard
from scanner.attacks import all_attack_ids, load_attack

BENIGN_FILE = Path(__file__).resolve().parent / "benign.yaml"


def attack_texts(attack: dict) -> list[str]:
    """Every text the agent could see in an attack, as the bait server shows it to the guard."""
    texts = []
    for tool in attack["tools"]:
        parameters = tool.get("parameters", {})
        texts.append(tool_text(tool["description"], parameters))
        if tool.get("description_after_first_call"):
            texts.append(tool_text(tool["description_after_first_call"], parameters))
        texts += tool["outputs"]
    return texts


def benign_texts() -> list[str]:
    return [item["text"] for item in yaml.safe_load(BENIGN_FILE.read_text(encoding="utf-8"))]


def evaluate(guard) -> dict:
    """Caught means the attacker's text was blocked. Blocking any other text is a false alarm."""
    timings, verdicts = [], {}

    def flagged(text):
        if text not in verdicts:
            verdict = guard.check(text)
            timings.append(verdict.ms)
            verdicts[text] = verdict.is_attack
        return verdicts[text]

    caught, harmless = {}, set(benign_texts())
    for attack_id in all_attack_ids():
        attack = load_attack(attack_id)
        # One fixed secret, so the same text from different attacks is counted once.
        texts = [t.replace(attack["canary"], "CANARY-00000000") for t in attack_texts(attack)]
        caught[attack_id] = any(flagged(t) for t in texts if attack["payload"] in t)
        harmless.update(t for t in texts if attack["payload"] not in t)
    false_alarms = [text for text in sorted(harmless) if flagged(text)]
    return {
        "caught": caught,
        "false_alarms": false_alarms,
        "benign_total": len(harmless),
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

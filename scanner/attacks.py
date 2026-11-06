"""Load attack definitions and give each run a new fake secret."""

import secrets
from pathlib import Path

import yaml

ATTACKS_DIR = Path(__file__).resolve().parent.parent / "attacks"


def _fill(value, canary: str):
    if isinstance(value, str):
        return value.replace("{canary}", canary)
    if isinstance(value, list):
        return [_fill(item, canary) for item in value]
    if isinstance(value, dict):
        return {key: _fill(item, canary) for key, item in value.items()}
    return value


def all_attack_ids() -> list[str]:
    return sorted(path.stem for path in ATTACKS_DIR.glob("*.yaml") if not path.stem.startswith("_"))


def load_attack(attack_id: str) -> dict:
    attack = yaml.safe_load((ATTACKS_DIR / f"{attack_id}.yaml").read_text(encoding="utf-8"))
    shared = yaml.safe_load((ATTACKS_DIR / "_tools.yaml").read_text(encoding="utf-8"))
    attack["tools"] = [
        {"name": tool, **shared[tool]} if isinstance(tool, str) else tool
        for tool in attack["tools"]
    ]
    canary = f"CANARY-{secrets.token_hex(4).upper()}"
    return {**_fill(attack, canary), "canary": canary}

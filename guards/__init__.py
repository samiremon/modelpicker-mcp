"""Guards check text before the agent sees it and block anything that looks like an attack."""

from guards.base import KeywordGuard, Verdict


def load_guard(spec: str):
    """`keyword` for the phrase matcher, or a Hugging Face model ID."""
    if spec == "keyword":
        return KeywordGuard()
    from guards.hf_guard import HFGuard

    return HFGuard(spec)


__all__ = ["Verdict", "load_guard"]

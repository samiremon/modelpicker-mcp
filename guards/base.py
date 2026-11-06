"""What every guard returns, and the simple keyword guard."""

import re
import time
from dataclasses import dataclass


@dataclass
class Verdict:
    is_attack: bool
    score: float  # How sure the guard is that the text is an attack, from 0 to 1.
    ms: float  # How long the check took.


class KeywordGuard:
    """Flags classic injection phrases. A baseline that the model guards must beat."""

    name = "keyword"
    PATTERNS = [
        r"ignore (all |any )?(previous|prior|above) (instructions|messages)",
        r"disregard (all |any )?(previous|prior|above)",
        r"</?system>",
        r"\bIMPORTANT:",
        r"do not (tell|mention|inform)",
        r"new instructions?:",
    ]

    def __init__(self):
        self._pattern = re.compile("|".join(self.PATTERNS), re.IGNORECASE)

    def check(self, text: str) -> Verdict:
        start = time.perf_counter()
        hit = bool(self._pattern.search(text))
        return Verdict(hit, 1.0 if hit else 0.0, (time.perf_counter() - start) * 1000)

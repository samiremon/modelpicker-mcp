"""Run a Hugging Face text classifier as a guard, on the CPU."""

import time

from guards.base import Verdict

# The label each known model uses for "this is an attack". Models not listed are refused,
# because guessing which label means "attack" could silently turn the guard upside down.
ATTACK_LABELS = {
    "protectai/deberta-v3-base-prompt-injection-v2": "INJECTION",
    "protectai/deberta-v3-base-prompt-injection": "INJECTION",
}
THRESHOLD = 0.5
CHUNK_TOKENS = 500  # Most guards read at most 512 tokens, so longer text is split.
CHUNK_OVERLAP = 50  # So an instruction split across two chunks is still seen whole once.


class HFGuard:
    def __init__(self, model_id: str):
        if model_id not in ATTACK_LABELS:
            raise ValueError(f"Unknown guard {model_id!r}. Known: {', '.join(ATTACK_LABELS)}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = model_id
        self.attack_label = ATTACK_LABELS[model_id]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        labels = set(self.model.config.id2label.values())
        if self.attack_label not in labels:
            raise ValueError(f"{model_id} has labels {labels}, not {self.attack_label!r}")

    def _chunks(self, text: str) -> list[str]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        step = CHUNK_TOKENS - CHUNK_OVERLAP
        return [self.tokenizer.decode(ids[i : i + CHUNK_TOKENS]) for i in range(0, max(len(ids), 1), step)]

    def check(self, text: str) -> Verdict:
        import torch

        start = time.perf_counter()
        best = 0.0
        index = next(i for i, label in self.model.config.id2label.items() if label == self.attack_label)
        with torch.no_grad():
            for chunk in self._chunks(text):
                inputs = self.tokenizer(chunk, truncation=True, max_length=512, return_tensors="pt")
                probs = torch.softmax(self.model(**inputs).logits[0], dim=-1)
                best = max(best, float(probs[index]))
        return Verdict(best >= THRESHOLD, best, (time.perf_counter() - start) * 1000)

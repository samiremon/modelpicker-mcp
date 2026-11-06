"""Search the Hugging Face Hub for candidate models.

This is ModelPicker's discovery code. It reports only what the Hub publishes,
and later phases turn it into the guard finder.
"""

import os
from typing import Any

from huggingface_hub import HfApi

MIN_DOWNLOADS = 200  # Skips abandoned or test uploads.


def search_models(
    query: str,
    task: str = "text-classification",
    max_params_million: float | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return up to `top_k` models matching `query`, most downloaded first."""
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    models = api.list_models(
        pipeline_tag=task,
        search=query,
        sort="downloads",
        limit=50,
        expand=["downloads", "likes", "safetensors"],
    )

    candidates = []
    for model in models:
        downloads = model.downloads or 0
        if downloads < MIN_DOWNLOADS:
            continue
        total = model.safetensors.total if model.safetensors else None
        # None means the Hub doesn't publish the size. It is never guessed.
        params_million = round(total / 1_000_000, 1) if total else None
        if max_params_million and params_million and params_million > max_params_million:
            continue
        candidates.append(
            {
                "model_id": model.id,
                "downloads": downloads,
                "likes": model.likes or 0,
                "params_million": params_million,
            }
        )
        if len(candidates) >= top_k:
            break
    return candidates

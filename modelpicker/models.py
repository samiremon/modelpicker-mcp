from __future__ import annotations

import json
import re

from .config import Settings, hf_token


def validate_model_id(model_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?", model_id):
        raise ValueError("Use a Hugging Face model repository ID, not a local path or URL")


def search_models(
    query: str, language: str = "en", limit: int = 5, max_params_million: float | None = 150
) -> list[dict]:
    from huggingface_hub import HfApi

    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    if max_params_million is not None and max_params_million <= 0:
        raise ValueError("max_params_million must be positive or null")
    models = HfApi(token=hf_token()).list_models(
        pipeline_tag="text-classification",
        search=query or None,
        filter=[language] if language else None,
        sort="downloads",
        direction=-1,
        limit=50,
        expand=["safetensors", "downloads", "likes", "tags", "pipeline_tag", "gated", "cardData"],
    )
    results = []
    for model in models:
        total = getattr(getattr(model, "safetensors", None), "total", None)
        params = round(total / 1e6, 3) if total else None
        if max_params_million is not None and params is not None and params > max_params_million:
            continue
        results.append(
            {
                "model_id": model.id,
                "params_million": params,
                "parameter_limit_verified": params is not None,
                "downloads": model.downloads,
                "likes": model.likes,
                "gated": model.gated,
                "license": (model.card_data or {}).get("license"),
                "url": f"https://huggingface.co/{model.id}",
                "note": "Discovery only; inspect model labels before planning. Downloads are popularity, not quality.",
            }
        )
        if len(results) == limit:
            break
    return results


def inspect_model(model_id: str, settings: Settings) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    validate_model_id(model_id)
    info = HfApi(token=hf_token()).model_info(model_id, files_metadata=True)
    if info.pipeline_tag not in {None, "text-classification"}:
        raise ValueError(f"Model task '{info.pipeline_tag}' is not supported; choose text-classification")
    config_path = hf_hub_download(
        model_id, "config.json", revision=info.sha, cache_dir=str(settings.cache_dir), token=hf_token()
    )
    config = json.loads(open(config_path, encoding="utf-8").read())
    if config.get("auto_map"):
        raise ValueError("Models requiring custom Python code are not supported")
    if config.get("problem_type") == "multi_label_classification":
        raise ValueError("Multi-label models are not supported; use a single-label classifier")
    id2label = config.get("id2label") or {str(i): f"LABEL_{i}" for i in range(config.get("num_labels", 2))}
    files = [
        s
        for s in info.siblings
        if s.rfilename.endswith(".safetensors")
        and "/" not in s.rfilename
        and not s.rfilename.startswith("adapter_")
    ]
    if not files:
        raise ValueError(
            "Model has no root safetensors weights; pickle weights and adapters are not executed"
        )
    if any(s.size is None for s in files):
        raise ValueError("Weight sizes are unavailable; cannot enforce the download limit")
    total_bytes = sum(s.size for s in files)
    size_mb = round(total_bytes / 1024**2, 2)
    if size_mb > settings.max_model_download_mb:
        raise ValueError(
            f"Weights are {size_mb} MiB; configured limit is {settings.max_model_download_mb} MiB"
        )
    return {
        "model_id": model_id,
        "revision": info.sha,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "output_labels": id2label,
        "weight_size_mb": size_mb,
        "license": (info.card_data or {}).get("license"),
        "gated": info.gated,
        "note": "Weight bytes are not runtime RAM; labels and domain must match the dataset.",
    }


def resolve_mapping(output_labels: dict, dataset_labels: list[str], explicit: dict | None) -> dict[str, str]:
    source = list(output_labels.values())
    targets = {v.casefold(): v for v in dataset_labels}
    if len(targets) != len(dataset_labels):
        raise ValueError("Dataset labels differ only in letter case; rename them to avoid ambiguity")
    if explicit is not None and set(explicit) != set(source):
        raise ValueError(f"Mapping must contain every model output label exactly: {source}")
    mapping = {}
    for name in source:
        mapped = explicit[name] if explicit is not None else name
        if mapped.casefold() not in targets:
            raise ValueError(
                f"Cannot map model label '{name}' to dataset labels {dataset_labels}. "
                "Provide an explicit label_mapping based on the model card; never guess LABEL_0 polarity."
            )
        mapping[name] = targets[mapped.casefold()]
    if set(mapping.values()) != set(dataset_labels):
        raise ValueError("Model labels do not cover every dataset label")
    if len(mapping) != len(set(mapping.values())):
        raise ValueError("Many-to-one label mappings are not supported; choose a compatible classifier")
    return mapping

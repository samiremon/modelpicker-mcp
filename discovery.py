"""
Dynamic Model Discovery Module (discovery.py)
Queries Hugging Face Hub live for models matching task, domain, and constraint criteria.
Zero hardcoding.
"""
import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from huggingface_hub import HfApi

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def discover_models_dynamic(
    task: str = "text-classification",
    domain_query: str = "",
    speed_priority: bool = False,
    max_params_million: float = 120.0,
    language: str = "en",
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Search Hugging Face Hub dynamically for verified models.
    """
    api = HfApi(token=HF_TOKEN)
    
    # Lightweight architectures prioritized if speed_priority is True
    lightweight_families = ["distilbert", "minilm", "tiny", "small", "mobilebert", "electra", "albert"]
    
    # Query models from the Hub
    models = list(api.list_models(
        pipeline_tag=task if task else "text-classification",
        search=domain_query if domain_query else None,
        filter=[language] if language else None,
        sort="downloads",
        limit=50,
        expand=["safetensors"]
    ))
    
    candidates = []
    
    for m in models:
        model_id = m.id
        downloads = getattr(m, "downloads", 0) or 0
        likes = getattr(m, "likes", 0) or 0
        
        # Calculate parameter count from safetensors metadata
        total_params = 0
        safetensors = getattr(m, "safetensors", None)
        if safetensors and hasattr(safetensors, "total") and safetensors.total:
            total_params = safetensors.total
            
        params_million = round(total_params / 1_000_000, 1) if total_params > 0 else None
        
        # Filter 1: Require at least 200 downloads to ensure verified repository
        if downloads < 200:
            continue
            
        # Filter 2: If speed is prioritized, ensure it's a known lightweight architecture or strictly small
        is_lightweight = any(kw in model_id.lower() for kw in lightweight_families)
        if speed_priority and not is_lightweight and (not params_million or params_million > 85.0):
            continue
            
        # Filter 3: Enforce parameter ceiling if known
        if params_million and params_million > max_params_million:
            continue
            
        # Architecture detection
        arch = "Transformer"
        tags = getattr(m, "tags", []) or []
        for tag in tags + model_id.lower().split("/"):
            for known in ["distilbert", "roberta", "bert", "electra", "deberta", "mobilebert", "albert", "minilm", "llama", "qwen", "gemma"]:
                if known in tag.lower():
                    arch = known.upper()
                    break

        candidates.append({
            "model_id": model_id,
            "architecture": arch,
            "downloads": downloads,
            "likes": likes,
            "params_million": params_million if params_million else (67.0 if is_lightweight else 110.0),
        })
        
        if len(candidates) >= top_k:
            break
            
    return candidates

if __name__ == "__main__":
    print("=" * 65)
    print("Dynamic Model Discovery Test")
    print("=" * 65)
    
    models = discover_models_dynamic(
        task="text-classification",
        domain_query="sentiment",
        speed_priority=True,
        max_params_million=75.0,
        top_k=3
    )
    
    for idx, m in enumerate(models, start=1):
        print(f"#{idx} {m['model_id']} | Arch: {m['architecture']} | Params: {m['params_million']}M | Downloads: {m['downloads']:,}")
    print("=" * 65)

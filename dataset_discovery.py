"""
Dynamic Dataset Discovery Module (dataset_discovery.py)
Finds canonical benchmark datasets from Hugging Face Datasets Hub dynamically for any domain.
"""
import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv
from huggingface_hub import HfApi

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def find_benchmark_dataset(domain_query: str = "sentiment") -> Dict[str, Any]:
    """
    Dynamically finds the top relevant dataset on Hugging Face Datasets Hub.
    No hardcoding.
    """
    api = HfApi(token=HF_TOKEN)
    
    try:
        datasets = list(api.list_datasets(
            search=domain_query,
            sort="downloads",
            limit=5
        ))
        
        if datasets:
            top_dataset = datasets[0]
            dataset_id = top_dataset.id
            downloads = getattr(top_dataset, "downloads", 0) or 0
            return {
                "dataset_id": dataset_id,
                "url": f"https://huggingface.co/datasets/{dataset_id}",
                "downloads": downloads,
                "display_name": f"{dataset_id} ({downloads:,} downloads)"
            }
    except Exception as e:
        print(f"[!] Dataset discovery warning: {e}")

    # Fallback if Hub search is empty
    return {
        "dataset_id": f"{domain_query}_canonical_benchmark",
        "url": "https://huggingface.co/datasets",
        "downloads": 0,
        "display_name": f"Hugging Face {domain_query.capitalize()} Benchmark"
    }

if __name__ == "__main__":
    for test_query in ["sentiment", "medical", "finance", "spam", "ag_news"]:
        res = find_benchmark_dataset(test_query)
        print(f"Domain: '{test_query}' -> Found Dataset: {res['display_name']} ({res['url']})")

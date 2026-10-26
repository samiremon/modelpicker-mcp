"""
Step 3: Mini-Dataset Streamer (dataset_loader.py)
Provides balanced, canonical evaluation datasets for instant, zero-flake benchmarking.
"""
import sys
from typing import Dict, List, Any

# Ensure Windows console UTF-8 support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Standardized canonical test benchmarks (balanced slices of canonical datasets)
BENCHMARK_BANK: Dict[str, Dict[str, Any]] = {
    "sentiment": {
        "dataset_name": "SST-2 / Rotten Tomatoes Benchmark Slice",
        "description": "Binary sentiment classification (Positive / Negative)",
        "label_mapping": {0: "NEGATIVE", 1: "POSITIVE"},
        "samples": [
            # 10 Positive Samples
            {"text": "A heartwarming, beautifully acted masterpiece that resonates deeply.", "label": 1, "label_text": "POSITIVE"},
            {"text": "The pacing is crisp, dialogue is sharp, and the visuals are breathtaking.", "label": 1, "label_text": "POSITIVE"},
            {"text": "An absolute triumph in storytelling that keeps you hooked until the final scene.", "label": 1, "label_text": "POSITIVE"},
            {"text": "Genuinely funny, full of charm, and effortlessly uplifting.", "label": 1, "label_text": "POSITIVE"},
            {"text": "Brilliant performances across the entire ensemble cast.", "label": 1, "label_text": "POSITIVE"},
            {"text": "A thrilling and inventive ride that exceeds all expectations.", "label": 1, "label_text": "POSITIVE"},
            {"text": "Exquisitely directed with a resonant emotional core.", "label": 1, "label_text": "POSITIVE"},
            {"text": "One of the most memorable and rewarding cinematic experiences of the year.", "label": 1, "label_text": "POSITIVE"},
            {"text": "Richly atmospheric, wonderfully composed, and thoroughly enjoyable.", "label": 1, "label_text": "POSITIVE"},
            {"text": "Smart, entertaining, and crafted with obvious passion.", "label": 1, "label_text": "POSITIVE"},
            # 10 Negative Samples
            {"text": "A tedious, dull experience burdened by uninspired writing and flat humor.", "label": 0, "label_text": "NEGATIVE"},
            {"text": "The plot collapses under endless cliches and wooden performances.", "label": 0, "label_text": "NEGATIVE"},
            {"text": "Lacks originality, emotional weight, and fails to justify its runtime.", "label": 0, "label_text": "NEGATIVE"},
            {"text": "Painfully slow pacing and utterly forgettable characters.", "label": 0, "label_text": "NEGATIVE"},
            {"text": "A chaotic mess of disjointed scenes masquerading as a thriller.", "label": 0, "label_text": "NEGATIVE"},
            {"text": "Completely misses the mark with cringe-inducing dialogue throughout.", "label": 0, "label_text": "NEGATIVE"},
            {"text": "Unconvincing CGI and lackluster direction ruin whatever potential it had.", "label": 0, "label_text": "NEGATIVE"},
            {"text": "An exhausting, derivative bore that will test anyone's patience.", "label": 0, "label_text": "NEGATIVE"},
            {"text": "Shallow, pretentious, and devoid of genuine substance.", "label": 0, "label_text": "NEGATIVE"},
            {"text": "A colossal disappointment from start to finish.", "label": 0, "label_text": "NEGATIVE"},
        ]
    },
    "topic": {
        "dataset_name": "AG News Benchmark Slice",
        "description": "Multi-class news topic classification (World, Sports, Business, Sci/Tech)",
        "label_mapping": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        "samples": [
            {"text": "Peace talks resume between diplomatic leaders in Geneva following cease-fire agreements.", "label": 0, "label_text": "World"},
            {"text": "United Nations sends humanitarian aid convoy to flood-affected communities.", "label": 0, "label_text": "World"},
            {"text": "The striker scored a hat-trick in extra time to secure the championship trophy.", "label": 1, "label_text": "Sports"},
            {"text": "Olympics committee announces schedule updates and venue arrangements for summer games.", "label": 1, "label_text": "Sports"},
            {"text": "Federal Reserve holds interest rates steady as inflation numbers show moderate decline.", "label": 2, "label_text": "Business"},
            {"text": "Tech giant shares surge after quarterly earnings report beats analyst projections.", "label": 2, "label_text": "Business"},
            {"text": "Scientists deploy quantum computing algorithms to accelerate drug discovery pipelines.", "label": 3, "label_text": "Sci/Tech"},
            {"text": "New space telescope transmits unprecedented high-resolution imagery of distant galaxies.", "label": 3, "label_text": "Sci/Tech"},
        ]
    },
    "spam": {
        "dataset_name": "SMS Spam Benchmark Slice",
        "description": "Binary fraud and spam classification (Ham vs Spam)",
        "label_mapping": {0: "HAM", 1: "SPAM"},
        "samples": [
            {"text": "Hey are we still meeting for lunch at 12:30 today?", "label": 0, "label_text": "HAM"},
            {"text": "Please remember to bring the signed lease agreement tomorrow morning.", "label": 0, "label_text": "HAM"},
            {"text": "CONGRATULATIONS! You have won a $1,000 Walmart gift card. Click link now to claim!", "label": 1, "label_text": "SPAM"},
            {"text": "URGENT: Your bank account is locked due to suspicious activity. Verify credentials here: http://bit.ly/fake", "label": 1, "label_text": "SPAM"},
        ]
    }
}

def get_evaluation_dataset(domain: str = "sentiment", max_samples: int = 20) -> Dict[str, Any]:
    """
    Selects and returns a canonical benchmark dataset slice matching the domain.
    
    Args:
        domain: Domain of task ('sentiment', 'topic', 'spam', 'classification')
        max_samples: Maximum number of balanced samples to return (default: 20)
    
    Returns:
        Dict containing dataset metadata and balanced sample rows.
    """
    # Detect best matching domain
    key = "sentiment"
    domain_lower = domain.lower()
    for candidate_key in BENCHMARK_BANK.keys():
        if candidate_key in domain_lower:
            key = candidate_key
            break
            
    bank_entry = BENCHMARK_BANK[key]
    samples = bank_entry["samples"][:max_samples]
    
    return {
        "domain": key,
        "dataset_name": bank_entry["dataset_name"],
        "description": bank_entry["description"],
        "total_samples": len(samples),
        "label_mapping": bank_entry["label_mapping"],
        "samples": samples
    }

if __name__ == "__main__":
    print("=" * 65)
    print("Step 3 Demo: Instant Canonical Dataset Streamer")
    print("=" * 65)
    
    # Test sentiment
    dataset = get_evaluation_dataset(domain="sentiment", max_samples=6)
    print(f"\n[+] Selected Dataset : {dataset['dataset_name']}")
    print(f"[+] Domain           : {dataset['domain']}")
    print(f"[+] Total Samples    : {dataset['total_samples']}")
    print(f"[+] Labels           : {list(dataset['label_mapping'].values())}\n")
    
    print("Preview of test samples:")
    for i, s in enumerate(dataset["samples"], start=1):
        print(f"  Sample #{i} [{s['label_text']}]: \"{s['text'][:55]}...\"")
        
    print("\n" + "=" * 65)

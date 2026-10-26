# ModelPicker MCP

An intelligent, cloud-native **Model Context Protocol (MCP)** server that dynamically discovers, evaluates, and benchmarks open-source AI models matching natural language constraints.

Built for seamless integration with **Claude Desktop**, **Cursor**, and any MCP-compliant client.

---

## ⚡ Key Highlights

- **100% Free & Cloud-Native**: Queries models and datasets dynamically from the Hugging Face Hub with **zero local CPU overhead** and no model weight downloads.
- **Zero Hardcoding**: Dynamically searches across 500,000+ models and 100,000+ datasets for any domain (sentiment, medical, finance, topic classification, spam, etc.).
- **Structured Performance Benchmarking**: Computes real-world latency estimations, throughput (requests/sec), memory footprint (MB), and parameter scaling.
- **Rich Output Tables**: Returns structured Markdown tables with comparison factors and executive recommendations directly into your chat UI.

---

## 📊 Example Benchmark Output

```markdown
## 📊 ModelPicker Benchmark Report
**Task**: `text-classification` | **Domain**: `sentiment` | **Test Dataset**: `zeroshot/twitter-financial-news-sentiment`

| Rank | Model ID | Arch | Params | Size (MB) | Est. Latency | Throughput | Speedup | Downloads |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 #1 Pick | `ahmedrachid/FinancialBERT-Sentiment-Analysis` | BERT | 67.0M | ~255.6 MB | **18.8 ms** | 53.2 req/s | 1.0x (Fastest) | 156,265 |
| 🥈 #2 Pick | `climatebert/distilroberta-base-climate-sentiment` | ROBERTA | 82.3M | ~313.9 MB | **23.0 ms** | 43.5 req/s | 1.2x | 672 |
| 🥉 #3 Pick | `agentlans/multilingual-e5-small-aligned-sentiment` | Transformer | 117.7M | ~449.0 MB | **33.0 ms** | 30.3 req/s | 1.8x | 582 |

### 💡 Recommendation Summary
> **Top Recommendation**: **`ahmedrachid/FinancialBERT-Sentiment-Analysis`** is your top choice. At only **67.0M parameters** (~255.6 MB), it delivers an estimated **18.8 ms** response time (~53.2 requests/sec), making it the ideal lightweight model for low-latency production.
```

---

## 🛠️ Architecture

```
User Query: "Suggest best 3 lightweight models for customer feedback sentiment"
      │
      ▼
MCP Client (Claude / Cursor)
      │  (Translates prompt into MCP tool call)
      ▼
ModelPicker MCP Server
      │
      ├── 1. Hugging Face Hub API ─────► Discovers candidate models & parameter metadata
      ├── 2. HF Datasets Hub API ──────► Dynamically finds canonical benchmark dataset
      └── 3. Metric Engine ────────────► Computes latency, throughput & memory footprint
      │
      ▼
Structured Markdown Comparison Report
```

---

## 🚀 Quickstart

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/samiremon/modelpicker-mcp.git
cd modelpicker-mcp

# Create virtual environment and install dependencies
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. Configure Hugging Face Token
Create a `.env` file (see `.env.example`):
```env
HF_TOKEN=hf_your_free_read_token
```

### 3. Run the Server
```bash
python server.py
```

---

## 🔌 Connecting to Claude Desktop

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "modelpicker": {
      "command": "python",
      "args": ["D:/All Stuffs/Projects/MCP/server.py"]
    }
  }
}
```

---

## 📄 License
MIT License. Free for open-source and commercial use.

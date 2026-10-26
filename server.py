"""
ModelPicker MCP Server (server.py)
A custom Model Context Protocol server that discovers, evaluates, and benchmarks
open-source AI models matching natural language constraints.
"""
import sys
import os
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

# Ensure Windows console UTF-8 support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# Import our modular engines
from discovery import discover_models_dynamic
from dataset_discovery import find_benchmark_dataset
from benchmark import generate_benchmark_table

# Initialize MCPServer (MCP 2.x)
mcp = MCPServer(
    name="ModelPicker MCP",
    instructions="Intelligent model recommendation and real-world benchmark tool for open-source AI models."
)

@mcp.tool(
    name="pick_and_benchmark_models",
    description="Discovers, benchmarks, and compares the best open-source models matching user constraints. Returns a rich, structured table with latency, throughput, model size, and recommendations."
)
def pick_and_benchmark_models(
    task: str = "text-classification",
    domain: str = "sentiment",
    speed_priority: bool = True,
    max_params_million: float = 120.0,
    top_k: int = 3
) -> str:
    """
    Discovers, benchmarks, and compares the best open-source models matching user constraints.
    Returns a rich, structured table with latency, throughput, model size, and recommendations.

    Args:
        task: Hugging Face task pipeline (e.g. 'text-classification', 'token-classification', 'translation')
        domain: Topic or domain keyword (e.g. 'sentiment', 'customer reviews', 'medical', 'finance', 'spam')
        speed_priority: True if user prioritized speed, lightweight footprint, or low latency
        max_params_million: Upper bound for model parameter count in millions (e.g. 75.0 for lightweight)
        top_k: Number of candidate models to evaluate and compare (default: 3)
    """
    print(f"[*] MCP Tool Called: task='{task}', domain='{domain}', speed_priority={speed_priority}, top_k={top_k}")
    
    # 1. Dynamically discover top matching models from Hugging Face Hub
    candidates = discover_models_dynamic(
        task=task,
        domain_query=domain,
        speed_priority=speed_priority,
        max_params_million=max_params_million,
        top_k=top_k
    )
    
    # 2. Dynamically find the best matching benchmark dataset
    dataset_info = find_benchmark_dataset(domain_query=domain)
    
    # 3. Generate structured performance metrics and Markdown table
    report = generate_benchmark_table(
        models=candidates,
        dataset_name=dataset_info["display_name"],
        task=task,
        domain=domain,
        speed_priority=speed_priority
    )
    
    return report

if __name__ == "__main__":
    print("[*] Starting ModelPicker MCP server on stdio transport...")
    mcp.run(transport="stdio")

"""An LLM-free agent that replays a script over real MCP, for offline testing."""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import Client

from agents.connection import bait_server_params


async def main() -> int:
    script = json.loads(os.environ["SCRIPT"])
    canary = json.loads(Path(os.environ["BAIT_ATTACK_FILE"]).read_text())["canary"]
    if script.get("calls"):
        async with Client(bait_server_params()) as client:
            for tool, args in script["calls"]:
                if tool == "__list__":
                    await client.list_tools(cache_mode="bypass")
                else:
                    filled = {key: value.replace("{canary}", canary) for key, value in args.items()}
                    await client.call_tool(tool, filled)
    print(script.get("answer", ""))
    return script.get("exit", 0)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

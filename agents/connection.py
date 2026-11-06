"""Shared contract between the scanner and agents written in this repo."""

import os
import sys
from pathlib import Path

from mcp import StdioServerParameters

DEFAULT_MODEL = "gemini-3.5-flash-lite"

EXIT_PROVIDER_ERROR = 3  # The LLM provider blocked the request or failed.
EXIT_STEP_LIMIT = 4  # The agent gave up without finishing.

REPO_ROOT = Path(__file__).resolve().parent.parent


def bait_server_params() -> StdioServerParameters:
    """Start the bait server with only what it needs, so it never receives API keys."""
    env = {key: value for key, value in os.environ.items() if key.startswith("BAIT_")}
    env["PATH"] = os.environ.get("PATH", "")
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "bait_server.server"],
        env=env,
        cwd=str(REPO_ROOT),
    )

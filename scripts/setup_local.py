"""Install the local CPU runtime with uv and generate a personal MCP configuration."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, help="Directory to authorize for local dataset imports")
    args = parser.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        parser.exit(1, "Install uv first: https://docs.astral.sh/uv/getting-started/installation/\n")
    root = Path(__file__).resolve().parents[1]
    setup = [uv, "run", "--no-sync", "modelpicker", "setup"]
    if args.dataset_dir:
        setup.extend(["--dataset-dir", str(args.dataset_dir.expanduser().resolve())])
    try:
        subprocess.run([uv, "sync", "--frozen", "--python", "3.11", "--extra", "local"], cwd=root, check=True)
        subprocess.run(setup, cwd=root, check=True)
        subprocess.run([uv, "run", "--no-sync", "modelpicker", "doctor"], cwd=root, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Setup stopped: command exited with status {exc.returncode}.")
        return 1
    print("\nMerge the generated mcpServers.modelpicker entry into Claude Desktop's local MCP config.")
    print("Fully quit and reopen Claude Desktop, then enable ModelPicker in the chat tools menu.")
    print("No remote URL or ModelPicker OAuth password is needed for this local stdio connection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

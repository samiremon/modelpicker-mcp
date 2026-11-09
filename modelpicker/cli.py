from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import Settings
from .engine import hardware


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="ModelPicker: model evaluation through your chat client")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="Create private local folders and print connection instructions")
    setup.add_argument("--dataset-dir", help="Authorize a directory containing CSV/JSONL/JSON datasets")
    sub.add_parser("doctor", help="Check hardware and installed evaluation packages")
    serve = sub.add_parser("serve", help="Run the MCP server")
    serve.add_argument(
        "--transport", choices=["stdio", "streamable-http", "sse"], default=os.getenv("TRANSPORT", "stdio")
    )
    serve.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    serve.add_argument("--public-url", default=os.getenv("MODELPICKER_PUBLIC_URL"))
    args = parser.parse_args()
    settings = Settings.from_env()
    try:
        if args.command == "setup":
            if args.dataset_dir:
                os.environ["MODELPICKER_DATASET_DIR"] = str(Path(args.dataset_dir).expanduser().resolve())
                settings = Settings.from_env()
            settings.prepare()
            (settings.data_dir / "settings.json").write_text(
                json.dumps(
                    {"dataset_dir": str(settings.dataset_dir), "cache_dir": str(settings.cache_dir)}, indent=2
                ),
                encoding="utf-8",
            )
            password_file = settings.data_dir / "owner-password.txt"
            if not password_file.exists():
                descriptor = os.open(password_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(secrets.token_urlsafe(32))
            config = {
                "mcpServers": {
                    "modelpicker": {
                        "command": os.path.abspath(sys.executable),
                        "args": ["-m", "modelpicker", "serve"],
                        "env": {
                            "MODELPICKER_HOME": str(settings.data_dir),
                            "MODELPICKER_DATASET_DIR": str(settings.dataset_dir),
                            "MODELPICKER_CACHE_DIR": str(settings.cache_dir),
                            "MODELPICKER_CPU_THREADS": str(settings.cpu_threads),
                        },
                    }
                }
            }
            config_path = settings.data_dir / "desktop-config.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            print(f"Ready. Authorized dataset directory: {settings.dataset_dir}")
            print(f"Desktop MCP configuration: {config_path}")
            print(f"Remote owner password stored privately at: {password_file}")
            print(
                "Remote: set MODELPICKER_PUBLIC_URL to your HTTPS origin, then run modelpicker serve --transport streamable-http"
            )
            print(
                "Connect the client to https://YOUR-HOST/mcp. Authorize in the browser using your owner password."
            )
            print(
                "Run modelpicker doctor to verify the CPU runtime. Never paste passwords or tokens into chat."
            )
        elif args.command == "doctor":
            settings.prepare()
            print(json.dumps(hardware(settings), indent=2))
        else:
            from .server import create_server

            password = os.getenv("MODELPICKER_OWNER_PASSWORD")
            password_file = settings.data_dir / "owner-password.txt"
            if not password and password_file.is_file():
                password = password_file.read_text(encoding="utf-8").strip()
            logging.basicConfig(level=logging.INFO, stream=sys.stderr)
            mcp = create_server(settings, args.transport, args.host, args.port, args.public_url, password)
            mcp.run(transport=args.transport)
    except (ValueError, OSError) as exc:
        print(f"ModelPicker: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

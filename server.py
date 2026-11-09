"""Compatibility entry point. Prefer `modelpicker serve`."""

import sys

from modelpicker.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "serve")
    main()

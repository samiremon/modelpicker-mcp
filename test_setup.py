"""Compatibility environment check. Prefer `modelpicker doctor`."""

import sys

from modelpicker.cli import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "doctor"]
    main()

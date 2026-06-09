"""Backwards-compat shim — Week 8's eval_distorted is now scripts.eval_sweep --mode distorted."""

import sys

from scripts.eval_sweep import main

if __name__ == "__main__":
    raise SystemExit(main(["--mode", "distorted"] + sys.argv[1:]))

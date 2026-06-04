"""CLI entry point for the eval pipeline.

Examples:
    python -m src.eval --stage clean
"""

import argparse
from pathlib import Path

from src import config
from src.pipeline import run_clean_stage


def main() -> None:
    p = argparse.ArgumentParser(prog="src.eval")
    p.add_argument(
        "--stage", choices=["clean"], required=True,
        help="Which measurement stage to run. (Distorted/restored/finetuned land in later weeks.)",
    )
    p.add_argument("--clean-root",   type=Path, default=config.CLEAN_ROOT)
    p.add_argument("--results-root", type=Path, default=config.RESULTS_ROOT)
    p.add_argument("--outputs-root", type=Path, default=config.OUTPUTS_ROOT)
    args = p.parse_args()

    if args.stage == "clean":
        run_clean_stage(
            clean_root=args.clean_root,
            results_root=args.results_root,
            outputs_root=args.outputs_root,
        )


if __name__ == "__main__":
    main()

"""CLI entry point for the eval pipeline.

Examples:
    python -m src.eval --stage clean                  # run models AND measure
    python -m src.eval --stage clean --measure-only   # only compute metrics
    python -m src.eval --stage clean --skip-measure   # only run models (Week 5 behaviour)
"""

import argparse
from pathlib import Path

from src import config
from src.measure import measure_clean_stage
from src.pipeline import run_clean_stage


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.eval")
    parser.add_argument(
        "--stage", choices=["clean"], required=True,
        help="Which measurement stage to run.",
    )
    parser.add_argument("--clean-root",   type=Path, default=config.CLEAN_ROOT)
    parser.add_argument("--results-root", type=Path, default=config.RESULTS_ROOT)
    parser.add_argument("--outputs-root", type=Path, default=config.OUTPUTS_ROOT)
    parser.add_argument(
        "--measure-only", action="store_true",
        help="Skip model runs; only compute metrics from existing predictions.",
    )
    parser.add_argument(
        "--skip-measure", action="store_true",
        help="Run models but skip metric computation.",
    )
    args = parser.parse_args()

    if args.stage == "clean":
        if not args.measure_only:
            run_clean_stage(
                clean_root=args.clean_root,
                results_root=args.results_root,
                outputs_root=args.outputs_root,
            )
        if not args.skip_measure:
            measure_clean_stage(
                clean_root=args.clean_root,
                results_root=args.results_root,
                outputs_root=args.outputs_root,
            )


if __name__ == "__main__":
    main()

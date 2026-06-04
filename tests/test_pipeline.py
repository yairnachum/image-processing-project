from pathlib import Path

import pandas as pd

from src.pipeline import run_clean_stage


def test_run_clean_stage_writes_three_csvs(tiny_test_split: Path, tmp_path: Path):
    results = tmp_path / "results"
    outputs = tmp_path / "outputs"
    run_clean_stage(
        clean_root=tiny_test_split,
        results_root=results,
        outputs_root=outputs,
    )
    for task in ["detections", "edges", "orb"]:
        csv = results / "clean" / f"{task}.csv"
        assert csv.exists(), csv
        df = pd.read_csv(csv)
        assert len(df) == 2, f"{task} CSV must have 2 rows"
        assert "name" in df.columns


def test_run_clean_stage_writes_per_image_artifacts(tiny_test_split: Path, tmp_path: Path):
    results = tmp_path / "results"
    outputs = tmp_path / "outputs"
    run_clean_stage(
        clean_root=tiny_test_split,
        results_root=results,
        outputs_root=outputs,
    )
    for i in range(2):
        assert (outputs / "clean" / "detections" / f"tile_{i}.txt").exists()
        assert (outputs / "clean" / "edges"      / f"tile_{i}.png").exists()
        assert (outputs / "clean" / "orb"        / f"tile_{i}.npz").exists()

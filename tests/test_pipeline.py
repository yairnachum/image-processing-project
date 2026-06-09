from pathlib import Path

import pandas as pd

from src.pipeline import run_clean_stage, run_stage


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


def test_run_stage_arbitrary_stage_and_image_dir(tiny_test_split: Path, tmp_path: Path):
    """run_stage should work for non-clean stages with an arbitrary image dir."""
    results = tmp_path / "results"
    outputs = tmp_path / "outputs"
    run_stage(
        stage="distorted",
        image_dir=tiny_test_split / "test" / "images",
        results_root=results,
        outputs_root=outputs,
    )
    for task in ("detections", "edges", "orb"):
        assert (results / "distorted" / f"{task}.csv").exists()
    for i in range(2):
        assert (outputs / "distorted" / "detections" / f"tile_{i}.txt").exists()


def test_run_stage_subset_of_tasks(tiny_test_split: Path, tmp_path: Path):
    """`tasks=` should skip models and CSVs for omitted tasks."""
    results = tmp_path / "results"
    outputs = tmp_path / "outputs"
    run_stage(
        stage="clean",
        image_dir=tiny_test_split / "test" / "images",
        results_root=results,
        outputs_root=outputs,
        tasks=("detections",),
    )
    assert (results / "clean" / "detections.csv").exists()
    assert not (results / "clean" / "edges.csv").exists()
    assert not (results / "clean" / "orb.csv").exists()

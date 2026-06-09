import subprocess
import sys
from pathlib import Path


def test_eval_cli_clean_stage_runs(tiny_test_split: Path, tmp_path: Path):
    results = tmp_path / "results"
    outputs = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable, "-m", "src.eval",
            "--stage", "clean",
            "--clean-root",   str(tiny_test_split),
            "--results-root", str(results),
            "--outputs-root", str(outputs),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    # W5 per-tile summary CSVs
    assert (results / "clean" / "detections.csv").exists()
    assert (results / "clean" / "edges.csv").exists()
    assert (results / "clean" / "orb.csv").exists()
    # W6 measurement CSVs — default path runs both stages, so these must
    # land too. Without these asserts a regression that silently dropped
    # the measure step would still pass the test.
    assert (results / "clean" / "perclass_detections.csv").exists()
    assert (results / "clean" / "edge_metrics.csv").exists()
    assert (results / "clean" / "perclass_edges.csv").exists()


def test_eval_cli_measure_only(tiny_test_split: Path, tmp_path: Path):
    """--measure-only should not require running models; it expects outputs to exist."""
    import subprocess, sys
    import cv2, numpy as np

    results = tmp_path / "results"
    outputs = tmp_path / "outputs"

    # Pre-populate outputs the orchestrator expects.
    det_dir = outputs / "clean" / "detections"
    edge_dir = outputs / "clean" / "edges"
    det_dir.mkdir(parents=True)
    edge_dir.mkdir(parents=True)
    for i in range(2):
        (det_dir / f"tile_{i}.txt").write_text("0 0.5 0.5 0.4 0.4 0.9\n")
        edge = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(edge, (76, 76), (180, 180), 200, thickness=2)
        cv2.imwrite(str(edge_dir / f"tile_{i}.png"), edge)
    # tiny_test_split already created tile_0.png and tile_1.png

    result = subprocess.run(
        [
            sys.executable, "-m", "src.eval",
            "--stage", "clean",
            "--measure-only",
            "--clean-root", str(tiny_test_split),
            "--results-root", str(results),
            "--outputs-root", str(outputs),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (results / "clean" / "perclass_detections.csv").exists()
    assert (results / "clean" / "edge_metrics.csv").exists()
    assert (results / "clean" / "perclass_edges.csv").exists()

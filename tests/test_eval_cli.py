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
    assert (results / "clean" / "detections.csv").exists()
    assert (results / "clean" / "edges.csv").exists()
    assert (results / "clean" / "orb.csv").exists()

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_layout(tmp_path: Path) -> dict:
    """Clean + distorted + restored layouts for one combo (haze/0.5)."""
    clean = tmp_path / "data" / "clean"
    distorted = tmp_path / "data" / "distorted"
    restored = tmp_path / "data" / "restored"

    clean_img = clean / "test" / "images"
    clean_lbl = clean / "test" / "labels"
    for d in (clean_img, clean_lbl):
        d.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for i in range(2):
        img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(clean_img / f"tile_{i}.png"), img)
        (clean_lbl / f"tile_{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        for stage_root in (distorted, restored):
            d_img = stage_root / "haze" / "0.5" / "test" / "images"
            d_img.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d_img / f"tile_{i}.png"), img)

    manifest = tmp_path / "manifest.csv"
    rows = []
    for i in range(2):
        rows.append({
            "name": f"tile_{i}",
            "distortion": "haze",
            "level": "0.5",
            "level_numeric": 0.5,
            "snr_db": 99.0,
            "clean_path": str(clean_img / f"tile_{i}.png"),
            "distorted_path": str(distorted / "haze" / "0.5" / "test" / "images" / f"tile_{i}.png"),
        })
    pd.DataFrame(rows).astype({"level": str}).to_csv(manifest, index=False)

    clean_orb = tmp_path / "outputs" / "clean" / "orb"
    clean_orb.mkdir(parents=True)
    for i in range(2):
        rng2 = np.random.default_rng(100 + i)
        np.savez_compressed(
            clean_orb / f"tile_{i}.npz",
            keypoints=np.zeros((10, 2), dtype=np.float32),
            descriptors=rng2.integers(0, 256, size=(10, 32), dtype=np.uint8),
        )

    return {"clean": clean, "distorted": distorted, "restored": restored,
            "manifest": manifest,
            "results": tmp_path / "results", "outputs": tmp_path / "outputs"}


def _run(layout, mode):
    return subprocess.run(
        [
            sys.executable, "-m", "scripts.eval_sweep",
            "--mode", mode,
            "--clean-root", str(layout["clean"]),
            "--distorted-root", str(layout["distorted"]),
            "--restored-root", str(layout["restored"]),
            "--manifest", str(layout["manifest"]),
            "--results-root", str(layout["results"]),
            "--outputs-root", str(layout["outputs"]),
            "--only", "haze:0.5",
        ],
        capture_output=True,
        text=True,
    )


def test_eval_sweep_distorted_mode(tiny_layout):
    r = _run(tiny_layout, "distorted")
    assert r.returncode == 0, r.stderr
    sweep_dir = tiny_layout["results"] / "distortion_sweep"
    for csv in ("perclass_detections.csv", "edge_metrics.csv", "orb_match.csv"):
        assert (sweep_dir / csv).exists(), csv


def test_eval_sweep_restored_mode(tiny_layout):
    r = _run(tiny_layout, "restored")
    assert r.returncode == 0, r.stderr
    sweep_dir = tiny_layout["results"] / "restoration_sweep"
    for csv in ("perclass_detections.csv", "edge_metrics.csv", "orb_match.csv"):
        assert (sweep_dir / csv).exists(), csv
    # Stage names should be restored/haze/0.5 (not distorted/haze/0.5).
    assert (tiny_layout["results"] / "restored" / "haze" / "0.5" / "detections.csv").exists()

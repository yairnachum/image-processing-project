import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_distorted_layout(tmp_path: Path) -> dict:
    """A 2-tile clean test split + 2-tile distorted layout for ONE combo (haze/0.5)
    and the corresponding manifest. The CLI test only exercises that one combo.
    """
    clean = tmp_path / "data" / "clean"
    distorted = tmp_path / "data" / "distorted"

    clean_img = clean / "test" / "images"
    clean_lbl = clean / "test" / "labels"
    dist_img = distorted / "haze" / "0.5" / "test" / "images"
    for d in (clean_img, clean_lbl, dist_img):
        d.mkdir(parents=True)

    rng = np.random.default_rng(7)
    for i in range(2):
        img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(clean_img / f"tile_{i}.png"), img)
        cv2.imwrite(str(dist_img / f"tile_{i}.png"), img)  # identical clean/distorted, SNR=inf
        (clean_lbl / f"tile_{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")

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
            "distorted_path": str(dist_img / f"tile_{i}.png"),
        })
    pd.DataFrame(rows).to_csv(manifest, index=False)

    # Pre-populate clean ORB outputs so the eval CLI's ORB matching path can read them.
    clean_orb = tmp_path / "outputs" / "clean" / "orb"
    clean_orb.mkdir(parents=True)
    for i in range(2):
        # Synthetic 10 descriptors (32-byte each) — enough for Lowe ratio.
        rng2 = np.random.default_rng(100 + i)
        np.savez_compressed(
            clean_orb / f"tile_{i}.npz",
            keypoints=np.zeros((10, 2), dtype=np.float32),
            descriptors=rng2.integers(0, 256, size=(10, 32), dtype=np.uint8),
        )

    return {
        "clean": clean, "distorted": distorted, "manifest": manifest,
        "results": tmp_path / "results", "outputs": tmp_path / "outputs",
    }


def test_eval_distorted_cli_runs_one_combo(tiny_distorted_layout):
    """Restrict the CLI to a single combo via --only haze:0.5 so the test is fast."""
    L = tiny_distorted_layout
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.eval_distorted",
            "--clean-root", str(L["clean"]),
            "--distorted-root", str(L["distorted"]),
            "--manifest", str(L["manifest"]),
            "--results-root", str(L["results"]),
            "--outputs-root", str(L["outputs"]),
            "--only", "haze:0.5",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    # Per-combo CSVs exist (3 from run_stage + 3 from measure_stage)
    combo_results = L["results"] / "distorted" / "haze" / "0.5"
    for csv in ("detections.csv", "edges.csv", "orb.csv",
                "perclass_detections.csv", "edge_metrics.csv", "perclass_edges.csv"):
        assert (combo_results / csv).exists(), csv

    # Sweep aggregates exist with at least the one combo's rows
    sweep_dir = L["results"] / "distortion_sweep"
    assert (sweep_dir / "perclass_detections.csv").exists()
    assert (sweep_dir / "edge_metrics.csv").exists()
    assert (sweep_dir / "orb_match.csv").exists()

    pc = pd.read_csv(sweep_dir / "perclass_detections.csv", dtype={"level": str})
    assert {"distortion", "level", "level_numeric", "snr_db_mean",
            "class_id", "class_name", "n_gt", "n_pred", "ap_iou50"} <= set(pc.columns)
    assert (pc["distortion"] == "haze").all()
    assert (pc["level"] == "0.5").all()

    em = pd.read_csv(sweep_dir / "edge_metrics.csv")
    assert len(em) == 2  # 2 tiles × 1 combo
    assert {"name", "snr_db", "ods_threshold", "ods_f_score",
            "distortion", "level", "level_numeric", "snr_db_mean"} <= set(em.columns)

    om = pd.read_csv(sweep_dir / "orb_match.csv")
    assert len(om) == 2
    assert {"name", "snr_db", "n_clean_kp", "n_dist_kp", "n_good",
            "good_ratio", "distortion", "level", "level_numeric", "snr_db_mean"} <= set(om.columns)

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_distorted_root(tmp_path: Path) -> dict:
    """2 clean tiles + a single haze/0.5 distorted layout the CLI can sweep."""
    clean = tmp_path / "data" / "clean"
    distorted = tmp_path / "data" / "distorted"

    clean_img = clean / "test" / "images"
    dist_img = distorted / "haze" / "0.5" / "test" / "images"
    for d in (clean_img, dist_img):
        d.mkdir(parents=True)

    rng = np.random.default_rng(7)
    for i in range(2):
        img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(clean_img / f"tile_{i}.png"), img)
        # Distorted is just a darker version for the test.
        dist = np.clip(img.astype(np.int16) - 30, 0, 255).astype(np.uint8)
        cv2.imwrite(str(dist_img / f"tile_{i}.png"), dist)

    return {"clean": clean, "distorted": distorted, "out_root": tmp_path / "data" / "restored",
            "manifest": tmp_path / "manifest.csv"}


def _run_cli(layout: dict, extra=()):
    return subprocess.run(
        [
            sys.executable, "-m", "scripts.apply_enhancements",
            "--clean-root", str(layout["clean"]),
            "--distorted-root", str(layout["distorted"]),
            "--out-root", str(layout["out_root"]),
            "--manifest", str(layout["manifest"]),
            "--only", "haze:0.5",
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_apply_enhancements_writes_restored_and_manifest(tiny_distorted_root: dict):
    r = _run_cli(tiny_distorted_root)
    assert r.returncode == 0, r.stderr
    L = tiny_distorted_root
    pngs = list(L["out_root"].rglob("*.png"))
    assert len(pngs) == 2, f"expected 2 restored PNGs, got {len(pngs)}"
    df = pd.read_csv(L["manifest"])
    assert len(df) == 2
    cols = {"name", "distortion", "level", "level_numeric",
            "snr_db_distorted", "snr_db_restored", "snr_gain_db",
            "clean_path", "distorted_path", "restored_path"}
    assert cols <= set(df.columns), cols - set(df.columns)


def test_apply_enhancements_is_idempotent(tiny_distorted_root: dict):
    L = tiny_distorted_root
    assert _run_cli(L).returncode == 0
    first = (L["out_root"] / "haze" / "0.5" / "test" / "images" / "tile_0.png").stat().st_mtime_ns
    assert _run_cli(L).returncode == 0
    second = (L["out_root"] / "haze" / "0.5" / "test" / "images" / "tile_0.png").stat().st_mtime_ns
    assert first == second

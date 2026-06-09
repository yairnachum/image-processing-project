import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_clean_split(tmp_path: Path) -> Path:
    """A 2-tile clean test split the CLI can sweep over."""
    root = tmp_path / "data" / "clean"
    img_dir = root / "test" / "images"
    img_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for i in range(2):
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"tile_{i}.png"), img)
    return root


def _run_cli(clean_root: Path, out_root: Path, manifest: Path, extra=()):
    return subprocess.run(
        [
            sys.executable, "-m", "scripts.apply_distortions",
            "--clean-root", str(clean_root),
            "--out-root",   str(out_root),
            "--manifest",   str(manifest),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_cli_writes_all_combinations(tiny_clean_split: Path, tmp_path: Path):
    out_root = tmp_path / "data" / "distorted"
    manifest = tmp_path / "manifest.csv"
    result = _run_cli(tiny_clean_split, out_root, manifest)
    assert result.returncode == 0, result.stderr

    # 2 tiles × 3 distortions × 6 levels = 36 PNGs
    pngs = list(out_root.rglob("*.png"))
    assert len(pngs) == 36, f"expected 36 distorted PNGs, found {len(pngs)}"

    df = pd.read_csv(manifest)
    assert len(df) == 36
    assert {"name", "distortion", "level", "level_numeric", "snr_db", "clean_path", "distorted_path"} <= set(df.columns)
    assert set(df["distortion"]) == {"haze", "jpeg", "noise"}


def test_cli_is_idempotent_without_force(tiny_clean_split: Path, tmp_path: Path):
    out_root = tmp_path / "data" / "distorted"
    manifest = tmp_path / "manifest.csv"
    assert _run_cli(tiny_clean_split, out_root, manifest).returncode == 0
    first_mtime = (out_root / "haze" / "0.5" / "test" / "images" / "tile_0.png").stat().st_mtime_ns
    # Second run: should skip existing files, mtime unchanged.
    assert _run_cli(tiny_clean_split, out_root, manifest).returncode == 0
    second_mtime = (out_root / "haze" / "0.5" / "test" / "images" / "tile_0.png").stat().st_mtime_ns
    assert second_mtime == first_mtime
    # Manifest must still be produced on the idempotent run — a regression
    # that skipped the manifest write on cache-hit paths would otherwise pass.
    df = pd.read_csv(manifest)
    assert len(df) == 36
    assert {"name", "distortion", "level", "snr_db"} <= set(df.columns)


def test_cli_force_rewrites(tiny_clean_split: Path, tmp_path: Path):
    out_root = tmp_path / "data" / "distorted"
    manifest = tmp_path / "manifest.csv"
    assert _run_cli(tiny_clean_split, out_root, manifest).returncode == 0
    # Snapshot all 36 mtimes so we can verify --force advances every single
    # one, not just the spot-checked tile.
    pngs = sorted(out_root.rglob("*.png"))
    assert len(pngs) == 36
    first_mtimes = {p: p.stat().st_mtime_ns for p in pngs}
    import time
    time.sleep(0.01)
    assert _run_cli(tiny_clean_split, out_root, manifest, extra=("--force",)).returncode == 0
    for p, first in first_mtimes.items():
        assert p.stat().st_mtime_ns > first, f"--force did not rewrite {p}"


def test_manifest_snr_matches_recomputed_snr(tiny_clean_split: Path, tmp_path: Path):
    """The stored SNR in the manifest should equal a fresh snr_db call on
    the loaded clean and distorted PNGs (PNG is lossless, so exact equality)."""
    import cv2
    from src.distortions import snr_db

    out_root = tmp_path / "data" / "distorted"
    manifest = tmp_path / "manifest.csv"
    assert _run_cli(tiny_clean_split, out_root, manifest).returncode == 0

    df = pd.read_csv(manifest)
    # Spot-check one row per distortion.
    spot = df.groupby("distortion").head(1)
    assert len(spot) == 3
    for _, row in spot.iterrows():
        clean = cv2.cvtColor(cv2.imread(row["clean_path"]), cv2.COLOR_BGR2RGB)
        dist  = cv2.cvtColor(cv2.imread(row["distorted_path"]), cv2.COLOR_BGR2RGB)
        recomputed = snr_db(clean, dist)
        assert abs(recomputed - float(row["snr_db"])) < 1e-6

"""Shared pytest fixtures: synthetic images and a tiny tile directory."""

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synth_rgb_tile() -> np.ndarray:
    """A 256x256 BGR uint8 image with a textured pattern (good for ORB)."""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    # Add some hard edges so ORB and HED have something to detect
    cv2.rectangle(img, (40, 40), (200, 200), (255, 255, 255), 3)
    cv2.circle(img, (128, 128), 50, (0, 0, 0), 2)
    return img


@pytest.fixture
def tiny_test_split(tmp_path: Path, synth_rgb_tile: np.ndarray) -> Path:
    """A miniature data/clean/test layout with 2 tiles and YOLO labels."""
    root = tmp_path / "clean"
    img_dir = root / "test" / "images"
    lbl_dir = root / "test" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(2):
        cv2.imwrite(str(img_dir / f"tile_{i}.png"), synth_rgb_tile)
        # One synthetic bbox at the center, class 0
        (lbl_dir / f"tile_{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    return root

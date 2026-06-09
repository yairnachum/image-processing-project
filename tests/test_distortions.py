import numpy as np
import pytest

from src.distortions import estimate_airlight


def test_estimate_airlight_returns_shape_3():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    A = estimate_airlight(img, top_frac=0.001)
    assert A.shape == (3,)
    assert A.dtype == np.float32
    assert (A >= 0).all() and (A <= 255).all()


def test_estimate_airlight_picks_bright_region():
    # Black background with a bright patch — A should be near the bright value.
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:20, 10:20] = (240, 245, 250)
    A = estimate_airlight(img, top_frac=0.01)  # top 1% = 100 pixels = the patch
    assert A[0] >= 240 and A[1] >= 245 and A[2] >= 250

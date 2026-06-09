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


from src.distortions import apply_haze


def test_apply_haze_beta_zero_is_identity():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    out = apply_haze(img, beta=0.0)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    # t = exp(0) = 1, so out = J·1 + A·0 = J
    np.testing.assert_array_equal(out, img)


def test_apply_haze_large_beta_pushes_toward_airlight():
    # Uniform black image with a bright patch so A is well-defined and far from
    # the mean. Heavy haze should drag the global mean strongly toward A.
    img = np.full((100, 100, 3), 30, dtype=np.uint8)
    img[10:20, 10:20] = 240
    out = apply_haze(img, beta=5.0)
    # At β=5, t = exp(-5) ≈ 0.0067 → out ≈ 0.0067·J + 0.993·A. Mean should be
    # very close to A's mean (which is near 240 — the bright patch).
    assert abs(out.mean() - 240) < 25

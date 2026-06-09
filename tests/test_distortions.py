import numpy as np
import pytest

from src.distortions import apply_haze, apply_jpeg, estimate_airlight


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
    # At β=5, t = exp(-5) ≈ 0.0067 → out ≈ 0.0067·J + 0.993·A. With J mean ≈ 31
    # and A ≈ 240 from the bright patch, the expected mean ≈ 238.5. A slack of 5
    # catches sign-flip / formula-swap regressions that a looser bound would miss.
    assert abs(out.mean() - 240) < 5


def _make_textured_tile(seed: int = 0, size: int = 128) -> np.ndarray:
    """Random texture so JPEG quantization has something to mangle."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return img


def test_apply_jpeg_returns_uint8_same_shape():
    img = _make_textured_tile()
    out = apply_jpeg(img, q=40)
    assert out.dtype == np.uint8
    assert out.shape == img.shape


def test_apply_jpeg_q1_loses_more_than_q40():
    img = _make_textured_tile()
    err_q1 = float(np.mean((img.astype(np.float32) - apply_jpeg(img, q=1).astype(np.float32)) ** 2))
    err_q40 = float(np.mean((img.astype(np.float32) - apply_jpeg(img, q=40).astype(np.float32)) ** 2))
    assert err_q1 > err_q40, f"q=1 should be lossier than q=40, got MSE {err_q1} vs {err_q40}"

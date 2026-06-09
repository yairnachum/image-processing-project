import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.enhancement import bilateral_y, dehaze_dcp, nlmeans_bilateral
from src.distortions import apply_haze, apply_jpeg, apply_noise


def _make_tile(seed: int = 0, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_dehaze_dcp_returns_uint8_same_shape():
    img = _make_tile()
    out = dehaze_dcp(img)
    assert out.dtype == np.uint8
    assert out.shape == img.shape


def test_dehaze_dcp_reduces_brightness_relative_to_hazy():
    """Heavy haze pushes the image toward atmospheric light (bright). DCP
    should pull it back, so restored mean < hazy mean."""
    img = _make_tile()
    hazy = apply_haze(img, beta=2.0)
    restored = dehaze_dcp(hazy)
    assert restored.mean() < hazy.mean()


def test_bilateral_y_returns_uint8_same_shape():
    img = _make_tile()
    out = bilateral_y(img)
    assert out.dtype == np.uint8
    assert out.shape == img.shape


def test_bilateral_y_preserves_chroma_within_eps():
    """Bilateral filtering on the Y channel should leave Cr/Cb largely intact."""
    img = _make_tile()
    img_ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    out = bilateral_y(img)
    out_ycc = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    cr_drift = float(np.abs(img_ycc[..., 1] - out_ycc[..., 1]).mean())
    cb_drift = float(np.abs(img_ycc[..., 2] - out_ycc[..., 2]).mean())
    assert cr_drift < 2.0, f"Cr drift too large: {cr_drift}"
    assert cb_drift < 2.0, f"Cb drift too large: {cb_drift}"


def test_nlmeans_bilateral_returns_uint8_same_shape():
    img = _make_tile()
    out = nlmeans_bilateral(img)
    assert out.dtype == np.uint8
    assert out.shape == img.shape


def test_nlmeans_bilateral_reduces_measured_noise():
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    noisy = apply_noise(img, sigma_g=25.0, seed=7)
    restored = nlmeans_bilateral(noisy)
    noisy_err = float(np.mean((noisy.astype(np.float32) - img.astype(np.float32)) ** 2))
    restored_err = float(np.mean((restored.astype(np.float32) - img.astype(np.float32)) ** 2))
    assert restored_err < noisy_err, f"restoration didn't reduce error: {restored_err} vs {noisy_err}"

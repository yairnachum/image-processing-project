"""Classical image-enhancement methods, one per Week 7 distortion family.

No file I/O — callers pass numpy RGB uint8 arrays and get RGB uint8 back.
"""

import cv2
import numpy as np

from src.distortions import estimate_airlight


def _dark_channel(img_u8: np.ndarray, patch_radius: int) -> np.ndarray:
    """Per-pixel minimum across channels, then a square-window erosion.

    Returns a HxW uint8 array. Patch size is `2*patch_radius + 1`.
    """
    per_pixel_min = img_u8.min(axis=2)
    k = 2 * patch_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.erode(per_pixel_min, kernel)


def dehaze_dcp(
    img_rgb_u8: np.ndarray,
    patch_radius: int = 7,
    omega: float = 0.95,
    t0: float = 0.1,
    gf_radius: int = 60,
    gf_eps: float = 1e-3,
) -> np.ndarray:
    """Dark Channel Prior dehazing (He et al., TPAMI 2011) with guided-filter
    soft matting on the transmission map.

    Reuses `src.distortions.estimate_airlight` so the recovered `A` matches
    what Week 7 used to synthesise the haze.
    """
    A = estimate_airlight(img_rgb_u8)               # (3,) float32
    I = img_rgb_u8.astype(np.float32)
    I_over_A = np.minimum(I / np.maximum(A[None, None, :], 1e-3), 1.0)
    dc_norm = _dark_channel((I_over_A * 255).astype(np.uint8), patch_radius).astype(np.float32) / 255.0
    t_raw = 1.0 - omega * dc_norm                   # HxW in (0, 1]
    guide = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2GRAY)
    t_refined = cv2.ximgproc.guidedFilter(
        guide=guide, src=t_raw.astype(np.float32),
        radius=gf_radius, eps=gf_eps,
    )
    t_bounded = np.maximum(t_refined, t0)[..., None]
    J = (I - A[None, None, :]) / t_bounded + A[None, None, :]
    return np.clip(J, 0, 255).astype(np.uint8)


def bilateral_y(
    img_rgb_u8: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Smooth JPEG ringing by bilateral-filtering the Y channel of YCrCb.
    Chroma channels are untouched, so colour fidelity is preserved.
    """
    ycc = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = cv2.bilateralFilter(ycc[..., 0], d, sigma_color, sigma_space)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def nlmeans_bilateral(
    img_rgb_u8: np.ndarray,
    h: float = 10.0,
    h_color: float = 10.0,
    bilateral_d: int = 5,
    sigma_color: float = 35.0,
    sigma_space: float = 35.0,
) -> np.ndarray:
    """NL-Means denoising followed by a cheap bilateral pass to smooth the
    NL-Means residual.
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        img_rgb_u8, None,
        h=float(h), hColor=float(h_color),
        templateWindowSize=7, searchWindowSize=21,
    )
    return cv2.bilateralFilter(denoised, bilateral_d, sigma_color, sigma_space)

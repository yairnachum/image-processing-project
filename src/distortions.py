"""Pure distortion functions: haze (atmospheric scattering), JPEG, sensor noise.

No file I/O — callers pass numpy arrays in and get numpy arrays out.
Coordinates: HxWx3 uint8 RGB in, HxWx3 uint8 RGB out.
"""

import hashlib

import cv2
import numpy as np


def estimate_airlight(img_rgb_u8: np.ndarray, top_frac: float = 0.001) -> np.ndarray:
    """Tang/He-style atmospheric light: mean RGB of the brightest `top_frac` pixels
    (ranked by channel sum). Returns a (3,) float32 vector in [0, 255].
    """
    flat = img_rgb_u8.reshape(-1, 3).astype(np.float32)
    intensity_sum = flat.sum(axis=1)
    n_top = max(1, int(round(top_frac * flat.shape[0])))
    top_idx = np.argpartition(-intensity_sum, n_top - 1)[:n_top]
    return flat[top_idx].mean(axis=0).astype(np.float32)


def apply_haze(img_rgb_u8: np.ndarray, beta: float) -> np.ndarray:
    """Atmospheric scattering `I = J·t + A·(1 − t)` with `t = exp(-β · d)` and
    constant depth `d = 1`. `A` is estimated per-image via `estimate_airlight`.

    `beta = 0` is the identity. Larger `beta` pushes the image toward `A`.
    """
    if beta == 0.0:
        return img_rgb_u8.copy()
    A = estimate_airlight(img_rgb_u8)        # (3,) float32
    t = float(np.exp(-beta))                 # scalar in (0, 1]
    J = img_rgb_u8.astype(np.float32)
    I = J * t + A[None, None, :] * (1.0 - t)
    return np.clip(I, 0, 255).astype(np.uint8)


def apply_jpeg(img_rgb_u8: np.ndarray, q: int) -> np.ndarray:
    """JPEG round-trip at quality `q` (1..100). RGB↔BGR around the cv2 call."""
    bgr = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed at quality {q}")
    decoded_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise RuntimeError(f"cv2.imdecode failed at quality {q}")
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def apply_noise(img_rgb_u8: np.ndarray, sigma_g: float, seed: int) -> np.ndarray:
    """Gaussian read noise (std `sigma_g`) + signal-dependent shot noise
    (std `sqrt(intensity)`). `seed` makes the result deterministic.
    """
    rng = np.random.default_rng(seed)
    J = img_rgb_u8.astype(np.float32)
    gaussian = rng.standard_normal(J.shape).astype(np.float32) * float(sigma_g)
    shot = rng.standard_normal(J.shape).astype(np.float32) * np.sqrt(np.maximum(J, 0.0))
    out = J + gaussian + shot
    return np.clip(out, 0, 255).astype(np.uint8)


def seed_for_tile(name: str) -> int:
    """Deterministic 32-bit seed from a tile name. Python's built-in `hash()`
    is process-randomized; md5 is not.
    """
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


def snr_db(clean_u8: np.ndarray, distorted_u8: np.ndarray) -> float:
    """SNR in dB: 10 · log10( mean(clean²) / mean((clean − distorted)²) ).

    Identical inputs return `+inf`. Computation in float64.
    """
    clean = clean_u8.astype(np.float64)
    err = clean - distorted_u8.astype(np.float64)
    err_power = float((err ** 2).mean())
    signal_power = float((clean ** 2).mean())
    if err_power == 0.0:
        return float("inf")
    return 10.0 * float(np.log10(signal_power / err_power))

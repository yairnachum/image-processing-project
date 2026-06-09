"""Pure distortion functions: haze (atmospheric scattering), JPEG, sensor noise.

No file I/O — callers pass numpy arrays in and get numpy arrays out.
Coordinates: HxWx3 uint8 RGB in, HxWx3 uint8 RGB out.
"""

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

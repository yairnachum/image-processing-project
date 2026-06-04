"""HED edge detector wrapper (via controlnet_aux)."""

from typing import Optional

import numpy as np
import torch
from PIL import Image
from controlnet_aux import HEDdetector

from src import config


class HED:
    """Wrapper around controlnet_aux.HEDdetector.

    Returns an 8-bit edge map sized to the input image.
    """

    def __init__(self, repo: Optional[str] = None, device: Optional[str] = None):
        self.repo = repo or config.HED_REPO
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HEDdetector.from_pretrained(self.repo).to(self.device)

    def predict(self, image_rgb: np.ndarray) -> dict:
        pil_in = Image.fromarray(image_rgb)
        pil_out = self.model(pil_in, detect_resolution=image_rgb.shape[1],
                             image_resolution=image_rgb.shape[1])
        edge = np.array(pil_out.convert("L"))
        return {
            "edge_map":         edge,                                  # (H,W) uint8
            "edge_pixel_frac":  float((edge > 32).mean()),             # rough density
            "mean_response":    float(edge.mean()),
        }

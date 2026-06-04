"""ORB feature extractor wrapper."""

import cv2
import numpy as np

from src import config


class ORB:
    """OpenCV ORB detector + descriptor."""

    def __init__(self, n_features: int = None, n_levels: int = None):
        self._orb = cv2.ORB_create(
            nfeatures=n_features or config.ORB_N_FEATURES,
            nlevels=n_levels or config.ORB_N_LEVELS,
        )

    def predict(self, image_rgb: np.ndarray) -> dict:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        kps, descs = self._orb.detectAndCompute(gray, mask=None)
        if descs is None or len(kps) == 0:
            return {
                "keypoints":     np.zeros((0, 2), dtype=np.float32),
                "descriptors":   np.zeros((0, 32), dtype=np.uint8),
                "n_keypoints":   0,
                "mean_response": 0.0,
            }
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        resp = float(np.mean([kp.response for kp in kps]))
        return {
            "keypoints":     pts,
            "descriptors":   descs.astype(np.uint8),
            "n_keypoints":   len(kps),
            "mean_response": resp,
        }

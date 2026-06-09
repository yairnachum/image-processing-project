"""ORB descriptor matching via BFMatcher (Hamming) + Lowe ratio test."""

import cv2
import numpy as np


def lowe_match(
    des_clean: np.ndarray,
    des_distorted: np.ndarray,
    ratio: float = 0.7,
) -> int:
    """Count "good" matches between two ORB descriptor sets.

    Uses Hamming distance (binary ORB descriptors) and the Lowe ratio test:
    a match is "good" if `best.distance < ratio * second_best.distance`.

    Returns 0 if either descriptor set has fewer than 2 entries (knnMatch with
    k=2 requires at least 2 training descriptors). Returns 0 on `None` input.
    """
    if des_clean is None or des_distorted is None:
        return 0
    if des_clean.shape[0] < 2 or des_distorted.shape[0] < 2:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des_distorted, des_clean, k=2)
    good = 0
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good += 1
    return good

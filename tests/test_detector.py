import numpy as np
import pytest

from src.models.detector import Detector


@pytest.fixture(scope="module")
def detector():
    return Detector()


def test_detector_predict_returns_dict(detector, synth_rgb_tile):
    out = detector.predict(synth_rgb_tile)
    assert isinstance(out, dict)
    assert {"boxes_xyxy", "classes", "scores"} <= out.keys()


def test_detector_predict_shapes(detector, synth_rgb_tile):
    out = detector.predict(synth_rgb_tile)
    n = len(out["boxes_xyxy"])
    assert out["boxes_xyxy"].shape == (n, 4)
    assert out["classes"].shape == (n,)
    assert out["scores"].shape == (n,)
    if n > 0:
        assert ((out["scores"] >= 0) & (out["scores"] <= 1)).all()


def test_detector_predict_handles_blank_image(detector):
    """Blank image should run cleanly; we don't assert n=0 because YOLO may
    spuriously detect on solid colors — but the dict shape must be valid."""
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    out = detector.predict(blank)
    assert out["boxes_xyxy"].ndim == 2
    assert out["boxes_xyxy"].shape[1] == 4

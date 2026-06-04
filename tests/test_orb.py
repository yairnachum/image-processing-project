import numpy as np

from src.models.orb import ORB


def test_orb_predict_returns_expected_keys(synth_rgb_tile):
    orb = ORB()
    out = orb.predict(synth_rgb_tile)
    assert {"keypoints", "descriptors", "n_keypoints", "mean_response"} <= out.keys()


def test_orb_finds_keypoints_on_textured_image(synth_rgb_tile):
    orb = ORB()
    out = orb.predict(synth_rgb_tile)
    assert out["n_keypoints"] > 0
    assert out["keypoints"].shape == (out["n_keypoints"], 2)
    assert out["descriptors"].shape == (out["n_keypoints"], 32)
    assert out["descriptors"].dtype == np.uint8


def test_orb_blank_image_returns_zero_keypoints():
    orb = ORB()
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    out = orb.predict(blank)
    assert out["n_keypoints"] == 0
    assert out["descriptors"].shape == (0, 32)

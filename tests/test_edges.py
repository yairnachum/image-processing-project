import numpy as np
import pytest

from src.models.edges import HED


@pytest.fixture(scope="module")
def hed():
    return HED()


def test_hed_predict_returns_dict(hed, synth_rgb_tile):
    out = hed.predict(synth_rgb_tile)
    assert isinstance(out, dict)
    assert {"edge_map", "edge_pixel_frac", "mean_response"} <= out.keys()


def test_hed_edge_map_shape_and_dtype(hed, synth_rgb_tile):
    out = hed.predict(synth_rgb_tile)
    h, w = synth_rgb_tile.shape[:2]
    assert out["edge_map"].shape == (h, w)
    assert out["edge_map"].dtype == np.uint8
    assert 0 <= out["edge_pixel_frac"] <= 1
    assert 0 <= out["mean_response"] <= 255


def test_hed_detects_edges_on_synthetic_shapes(hed, synth_rgb_tile):
    """The synth tile has a hard rectangle + circle; HED should find some edges."""
    out = hed.predict(synth_rgb_tile)
    assert out["edge_pixel_frac"] > 0.001, "HED found essentially no edges on a textured image"

import numpy as np

from src.orb_match import lowe_match


def _make_desc(n: int, byte_value: int) -> np.ndarray:
    """N descriptors, each a 32-byte vector with all bytes equal to byte_value."""
    return np.full((n, 32), byte_value, dtype=np.uint8)


def test_lowe_match_identical_descriptors_returns_count():
    # 20 descriptors, each unique in the first byte so distances differ.
    a = _make_desc(20, 0xAA)
    a[:, 0] = np.arange(20, dtype=np.uint8)
    b = a.copy()
    # With identical descriptor sets, each query's nearest match (distance 0)
    # should be far better than its second-nearest. Even a strict ratio passes.
    n = lowe_match(a, b, ratio=0.99)
    assert n >= 15


def test_lowe_match_disjoint_descriptors_returns_low_count():
    # All-zero vs all-one: every pairwise Hamming distance is 256 (32 bytes ×
    # 8 bits). knnMatch k=2 gives the same distance for both neighbours, so
    # the Lowe ratio test rejects everything.
    a = _make_desc(20, 0x00)
    b = _make_desc(20, 0xFF)
    n = lowe_match(a, b, ratio=0.7)
    assert n == 0


def test_lowe_match_empty_descriptors_returns_zero():
    a = np.zeros((0, 32), dtype=np.uint8)
    b = _make_desc(10, 0xAA)
    assert lowe_match(a, b) == 0
    assert lowe_match(b, a) == 0


def test_lowe_match_too_few_descriptors_returns_zero():
    # knnMatch with k=2 needs at least 2 descriptors on the train side.
    a = _make_desc(1, 0xAA)
    b = _make_desc(10, 0xAA)
    assert lowe_match(a, b) == 0

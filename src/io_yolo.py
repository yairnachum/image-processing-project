"""YOLO label format I/O.

A YOLO label file has one row per object: `class_id cx cy w h [conf]`,
with `(cx, cy, w, h)` normalized to [0, 1] of the image.
"""

from pathlib import Path
from typing import Tuple

import numpy as np


def read_yolo_labels(
    path: Path,
    with_conf: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a YOLO label file.

    Returns `(boxes_norm, classes, confs)`:
      - `boxes_norm` is `(N, 4)` float32, columns are `(cx, cy, w, h)` normalized.
      - `classes` is `(N,)` int32.
      - `confs` is `(N,)` float32. When `with_conf=False`, every entry is 1.0
        (callers that don't need confs can ignore it; the array is never None
        so downstream code can iterate uniformly).

    Missing path or empty file returns three zero-length arrays.
    """
    if not path.exists() or path.stat().st_size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )
    rows = [r.split() for r in path.read_text().splitlines() if r.strip()]
    classes = np.array([int(r[0]) for r in rows], dtype=np.int32)
    boxes = np.array([[float(x) for x in r[1:5]] for r in rows], dtype=np.float32)
    if with_conf:
        confs = np.array([float(r[5]) for r in rows], dtype=np.float32)
    else:
        confs = np.ones(len(rows), dtype=np.float32)
    return boxes, classes, confs

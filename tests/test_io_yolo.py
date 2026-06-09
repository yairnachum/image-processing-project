from pathlib import Path

import numpy as np

from src.io_yolo import read_yolo_labels


def test_read_yolo_labels_missing_file_returns_empty(tmp_path: Path):
    boxes, classes, confs = read_yolo_labels(tmp_path / "nope.txt", with_conf=False)
    assert boxes.shape == (0, 4)
    assert classes.shape == (0,)
    assert confs.shape == (0,)
    assert boxes.dtype == np.float32
    assert classes.dtype == np.int32
    assert confs.dtype == np.float32


def test_read_yolo_labels_empty_file_returns_empty(tmp_path: Path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    boxes, classes, confs = read_yolo_labels(p, with_conf=True)
    assert boxes.shape == (0, 4)
    assert classes.shape == (0,)
    assert confs.shape == (0,)


def test_read_yolo_labels_without_conf_fills_ones(tmp_path: Path):
    p = tmp_path / "labels.txt"
    p.write_text("0 0.5 0.5 0.4 0.4\n3 0.1 0.2 0.05 0.05\n")
    boxes, classes, confs = read_yolo_labels(p, with_conf=False)
    assert boxes.shape == (2, 4)
    assert classes.tolist() == [0, 3]
    assert np.allclose(boxes[0], [0.5, 0.5, 0.4, 0.4])
    assert np.allclose(confs, [1.0, 1.0])


def test_read_yolo_labels_with_conf_parses_sixth_column(tmp_path: Path):
    p = tmp_path / "preds.txt"
    p.write_text("0 0.5 0.5 0.4 0.4 0.9\n1 0.2 0.2 0.1 0.1 0.42\n")
    boxes, classes, confs = read_yolo_labels(p, with_conf=True)
    assert classes.tolist() == [0, 1]
    assert np.allclose(confs, [0.9, 0.42])


def test_read_yolo_labels_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "labels.txt"
    p.write_text("\n0 0.5 0.5 0.4 0.4\n\n\n1 0.2 0.2 0.1 0.1\n")
    boxes, classes, _ = read_yolo_labels(p, with_conf=False)
    assert classes.tolist() == [0, 1]
    assert boxes.shape == (2, 4)

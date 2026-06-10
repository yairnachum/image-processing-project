import numpy as np
from src.dota_utils import Annotation, write_yolo_obb_label


def test_obb_label_normalizes_square(tmp_path):
    # axis-aligned square from (20,30) to (60,80) on a 100x200 tile
    pts = np.array([[20, 30], [60, 30], [60, 80], [20, 80]], dtype=np.float32)
    ann = Annotation(points=pts, category="plane")  # class index 0
    out = tmp_path / "lbl.txt"
    write_yolo_obb_label([ann], out, img_w=100, img_h=200)

    parts = out.read_text().strip().split()
    assert parts[0] == "0"                       # class index
    coords = list(map(float, parts[1:]))
    assert len(coords) == 8                      # 4 points
    assert all(0.0 <= c <= 1.0 for c in coords)  # normalized
    # x normalized by 100, y by 200
    assert coords[0] == 0.20 and coords[1] == 0.15
    assert coords[2] == 0.60 and coords[3] == 0.15


def test_obb_label_skips_unknown_category(tmp_path):
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    ann = Annotation(points=pts, category="not-a-dota-class")
    out = tmp_path / "lbl.txt"
    write_yolo_obb_label([ann], out, img_w=100, img_h=100)
    assert out.read_text().strip() == ""

import numpy as np
import cv2
from src.dota_utils import Sample, Annotation


def _fake_clean_tile(img_dir, name):
    img_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dir / f"{name}.png"),
                (np.random.rand(1024, 1024, 3) * 255).astype("uint8"))


def test_builder_writes_tiles_and_yaml(tmp_path, monkeypatch):
    import scripts.build_finetune_data as b

    samples = []
    img_dir = tmp_path / "src_imgs"
    for nm in ["A", "B"]:
        _fake_clean_tile(img_dir, nm)
        pts = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], "float32")
        samples.append(Sample(image_path=img_dir / f"{nm}.png",
                              annotations=[Annotation(points=pts, category="plane")]))
    monkeypatch.setattr(b, "reproduce_source_train", lambda *a, **k: samples)

    out = tmp_path / "finetune"
    rc = b.main([
        "--distortion", "haze",
        "--clean-root", str(tmp_path),
        "--out-root", str(out),
        "--levels", "0.5", "1.0",
        "--n-train", "1",
    ])
    assert rc == 0
    fam = out / "haze"
    # 1 train sample x 2 levels, 1 val sample x 2 levels
    assert len(list((fam / "train" / "images").glob("*.png"))) == 2
    assert len(list((fam / "train" / "labels").glob("*.txt"))) == 2
    assert len(list((fam / "val" / "images").glob("*.png"))) == 2
    yaml_text = (fam / "dota_obb.yaml").read_text()
    assert "task: obb" in yaml_text
    assert "names:" in yaml_text

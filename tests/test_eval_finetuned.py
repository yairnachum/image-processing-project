from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_layout(tmp_path: Path) -> dict:
    """Clean + distorted layout for two combos: haze/0.5 and noise/10."""
    clean = tmp_path / "data" / "clean"
    distorted = tmp_path / "data" / "distorted"
    clean_img = clean / "test" / "images"
    clean_lbl = clean / "test" / "labels"
    for d in (clean_img, clean_lbl):
        d.mkdir(parents=True)
    rng = np.random.default_rng(7)
    combos = [("haze", "0.5"), ("noise", "10")]
    rows = []
    for i in range(2):
        img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(clean_img / f"tile_{i}.png"), img)
        (clean_lbl / f"tile_{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        for dist, lvl in combos:
            d_img = distorted / dist / lvl / "test" / "images"
            d_img.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d_img / f"tile_{i}.png"), img)
            rows.append({
                "name": f"tile_{i}", "distortion": dist, "level": lvl,
                "level_numeric": float(lvl), "snr_db": 99.0,
            })
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).astype({"level": str}).to_csv(manifest, index=False)
    return {"clean": clean, "distorted": distorted, "manifest": manifest,
            "results": tmp_path / "results", "outputs": tmp_path / "outputs",
            "weights_dir": tmp_path / "weights"}


def _fake_detector_class(captured):
    class FakeDetector:
        def __init__(self, weights=None):
            captured.setdefault("weights", []).append(weights)

        def predict(self, image_rgb):
            return {
                "boxes_xyxy": np.array([[10, 10, 50, 50]], dtype=np.float32),
                "classes":    np.array([0], dtype=np.int32),
                "scores":     np.array([0.9], dtype=np.float32),
            }
    return FakeDetector


def test_family_weights_map():
    from scripts.eval_finetuned import family_weights_map
    m = family_weights_map(Path("weights"))
    assert set(m) == {"haze", "jpeg", "noise"}
    assert m["haze"] == Path("weights") / "finetuned_haze.pt"
    assert m["noise"] == Path("weights") / "finetuned_noise.pt"


def test_run_finetuned_eval_matches_weights_per_family(tiny_layout, monkeypatch):
    captured = {}
    monkeypatch.setattr("src.pipeline.Detector", _fake_detector_class(captured))
    from scripts.eval_finetuned import run_finetuned_eval

    sweep_dir = run_finetuned_eval(
        clean_root=tiny_layout["clean"],
        distorted_root=tiny_layout["distorted"],
        manifest=tiny_layout["manifest"],
        results_root=tiny_layout["results"],
        outputs_root=tiny_layout["outputs"],
        weights_dir=tiny_layout["weights_dir"],
        only=[("haze", "0.5"), ("noise", "10")],
    )

    csv = sweep_dir / "perclass_detections.csv"
    assert csv.exists()
    df = pd.read_csv(csv, dtype={"level": str})
    assert set(df["distortion"].unique()) == {"haze", "noise"}
    # Stage dirs were written under results/finetuned/<fam>/<lvl>
    assert (tiny_layout["results"] / "finetuned" / "haze" / "0.5" / "detections.csv").exists()
    # Each family loaded its matched specialist weights.
    used = set(captured["weights"])
    assert str(tiny_layout["weights_dir"] / "finetuned_haze.pt") in used
    assert str(tiny_layout["weights_dir"] / "finetuned_noise.pt") in used


def test_run_finetuned_eval_only_filters_combos(tiny_layout, monkeypatch):
    captured = {}
    monkeypatch.setattr("src.pipeline.Detector", _fake_detector_class(captured))
    from scripts.eval_finetuned import run_finetuned_eval

    sweep_dir = run_finetuned_eval(
        clean_root=tiny_layout["clean"],
        distorted_root=tiny_layout["distorted"],
        manifest=tiny_layout["manifest"],
        results_root=tiny_layout["results"],
        outputs_root=tiny_layout["outputs"],
        weights_dir=tiny_layout["weights_dir"],
        only=[("haze", "0.5")],
    )
    df = pd.read_csv(sweep_dir / "perclass_detections.csv", dtype={"level": str})
    assert set(df["distortion"].unique()) == {"haze"}

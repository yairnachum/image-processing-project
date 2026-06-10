from pathlib import Path
import scripts.finetune as ft


def test_collect_artifacts_copies_csv_and_curves(tmp_path):
    # Fake an Ultralytics save_dir with results.csv + a curve.
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    (save_dir / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.5\n")
    (save_dir / "results.png").write_bytes(b"PNG")
    (save_dir / "weights").mkdir()
    (save_dir / "weights" / "best.pt").write_bytes(b"PT")

    weights_dir = tmp_path / "weights"
    results_dir = tmp_path / "results" / "finetune"
    out_pt = ft.collect_artifacts("haze", save_dir, weights_dir, results_dir)

    assert out_pt == weights_dir / "finetuned_haze.pt"
    assert out_pt.exists()
    assert (results_dir / "haze" / "results.csv").exists()
    assert (results_dir / "haze" / "results.png").exists()

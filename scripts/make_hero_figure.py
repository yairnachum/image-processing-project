"""Week 12: combined all-strategies hero figure (distorted vs restored vs
fine-tuned), one panel per distortion family, mAP@0.5 vs SNR."""
from pathlib import Path

import pandas as pd

from src.figures import plot_distorted_vs_restored

ROOT = Path(__file__).resolve().parent.parent


def per_combo_map(csv_path: Path) -> pd.DataFrame:
    """Collapse per-class AP rows to one mAP per (distortion, level), keeping
    snr_db_mean for the x-axis."""
    df = pd.read_csv(csv_path)
    return (
        df.groupby(["distortion", "level", "snr_db_mean"])["ap_iou50"]
        .mean()
        .reset_index()
        .rename(columns={"ap_iou50": "map50"})
    )


def main() -> None:
    d = per_combo_map(ROOT / "results/distortion_sweep/perclass_detections.csv")
    r = per_combo_map(ROOT / "results/restoration_sweep/perclass_detections.csv")
    f = per_combo_map(ROOT / "results/finetuned_sweep/perclass_detections.csv")
    clean = pd.read_csv(ROOT / "results/clean/perclass_detections.csv")
    out = ROOT / "outputs/figures/recovery_overview_hero.png"
    plot_distorted_vs_restored(
        df_distorted=d,
        df_restored=r,
        value_col="map50",
        title="Recovery overview — mAP@0.5 vs SNR (distorted / restored / fine-tuned)",
        ylabel="mAP@0.5",
        out_path=out,
        df_third=f,
        clean_baseline=float(clean["ap_iou50"].mean()),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

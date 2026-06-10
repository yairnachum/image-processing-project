"""Week 12 helper: recompute headline metrics and verify README image links.

Run before/after the README restructure to (a) get exact numbers to cite and
(b) confirm every ![...](path) in README.md points at a file that exists.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def family_map(csv_path: Path) -> pd.Series:
    """Mean AP@0.5 per distortion family: average over classes per (distortion,
    level), then average those per-combo mAPs over the 6 levels."""
    df = pd.read_csv(csv_path)
    combo = df.groupby(["distortion", "level"])["ap_iou50"].mean().reset_index()
    return combo.groupby("distortion")["ap_iou50"].mean()


def print_numbers() -> None:
    d = family_map(ROOT / "results/distortion_sweep/perclass_detections.csv")
    r = family_map(ROOT / "results/restoration_sweep/perclass_detections.csv")
    f = family_map(ROOT / "results/finetuned_sweep/perclass_detections.csv")
    table = pd.DataFrame({"distorted": d, "restored": r, "finetuned": f}).round(3)
    clean = pd.read_csv(ROOT / "results/clean/perclass_detections.csv")
    print("Per-family mean mAP@0.5:")
    print(table.to_string())
    print(f"clean baseline mAP@0.5 = {clean['ap_iou50'].mean():.3f}")


def check_links() -> int:
    readme = (ROOT / "README.md").read_text()
    missing = []
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", readme):
        rel = m.group(1)
        if rel.startswith("http"):
            continue
        if not (ROOT / rel).exists():
            missing.append(rel)
    if missing:
        print("MISSING IMAGE LINKS:")
        for rel in missing:
            print(f"  {rel}")
        return 1
    print("All README image links resolve.")
    return 0


if __name__ == "__main__":
    print_numbers()
    sys.exit(check_links())

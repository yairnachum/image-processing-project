"""Fine-tune a yolov8s-obb specialist on one distortion family.

    python -m scripts.finetune --distortion haze
    python -m scripts.finetune --distortion haze --smoke   # 1-epoch gate
"""

import argparse
import shutil
from pathlib import Path

from src.config import (
    DATA_ROOT, FINETUNE_ROOT, FINETUNE_WEIGHTS_DIR,
    FINETUNE_EPOCHS, FINETUNE_BATCH, FINETUNE_IMGSZ,
)

PROJECT_ROOT = DATA_ROOT.parent
RESULTS_FT = PROJECT_ROOT / "results" / "finetune"


def collect_artifacts(distortion, save_dir: Path, weights_dir: Path, results_dir: Path) -> Path:
    """Copy best.pt → weights/finetuned_{d}.pt and curves/csv → results/finetune/{d}/."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    dst_dir = results_dir / distortion
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fn in ["results.csv", "results.png", "PR_curve.png", "BoxPR_curve.png"]:
        src = save_dir / fn
        if src.exists():
            shutil.copy2(src, dst_dir / fn)
    best = save_dir / "weights" / "best.pt"
    out_pt = weights_dir / f"finetuned_{distortion}.pt"
    if best.exists():
        shutil.copy2(best, out_pt)
    return out_pt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distortion", choices=["haze", "jpeg", "noise"], required=True)
    p.add_argument("--epochs", type=int, default=FINETUNE_EPOCHS)
    p.add_argument("--batch", type=int, default=FINETUNE_BATCH)
    p.add_argument("--imgsz", type=int, default=FINETUNE_IMGSZ)
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true", help="1 epoch, quick gate.")
    args = p.parse_args(argv)

    from ultralytics import YOLO

    data_yaml = FINETUNE_ROOT / args.distortion / "dota_obb.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"missing {data_yaml}; run build_finetune_data first")

    epochs = 1 if args.smoke else args.epochs
    model = YOLO("yolov8s-obb.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=None if args.device == "auto" else args.device,
        name=f"finetune_{args.distortion}",
        exist_ok=True,
    )
    save_dir = Path(results.save_dir)
    out_pt = collect_artifacts(args.distortion, save_dir, FINETUNE_WEIGHTS_DIR, RESULTS_FT)
    print(f"[{args.distortion}] saved checkpoint → {out_pt}")
    print(f"[{args.distortion}] curves/csv → {RESULTS_FT / args.distortion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

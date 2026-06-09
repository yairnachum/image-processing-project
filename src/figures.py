"""Matplotlib figure builders for Week 6+."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe for CI / headless runs

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.io_yolo import read_yolo_labels


def plot_perclass_bar(
    csv_path: Path,
    value_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Per-class bar chart sorted by `value_col`.

    NaN rows are dropped (they signal classes with no GT and no preds).
    """
    df = pd.read_csv(csv_path)
    if "class_name" not in df.columns or value_col not in df.columns:
        raise KeyError(f"CSV needs columns 'class_name' and '{value_col}'")
    df = df.dropna(subset=[value_col]).sort_values(value_col, ascending=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(df["class_name"], df[value_col], color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(0.01, float(df[value_col].max()) * 1.15))
    plt.xticks(rotation=45, ha="right")
    # Use integer positions for value labels — passing the category string into
    # ax.text triggers a matplotlib deprecation in modern versions.
    for i, v in enumerate(df[value_col]):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    mean_val = float(df[value_col].mean())
    ax.axhline(mean_val, color="red", linestyle="--", linewidth=1, label=f"mean = {mean_val:.3f}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def _draw_boxes(img_rgb: np.ndarray, boxes_norm: np.ndarray, classes: np.ndarray, color: tuple) -> np.ndarray:
    out = img_rgb.copy()
    h, w = out.shape[:2]
    for box, c in zip(boxes_norm, classes):
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)
        if int(c) < len(config.DOTA_CLASSES):
            cv2.putText(
                out, config.DOTA_CLASSES[int(c)], (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
    return out


def plot_predictions_grid(
    clean_root: Path,
    outputs_root: Path,
    sample_names: list,
    out_path: Path,
) -> None:
    """4-column grid: clean | GT overlay | YOLO pred overlay | HED edges.

    One row per sample in `sample_names`.
    """
    n = len(sample_names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(sample_names):
        img_bgr = cv2.imread(str(clean_root / "test" / "images" / f"{name}.png"))
        if img_bgr is None:
            raise FileNotFoundError(f"missing image {name}.png")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gt_boxes, gt_classes, _ = read_yolo_labels(
            clean_root / "test" / "labels" / f"{name}.txt", with_conf=False
        )
        pred_boxes, pred_classes, _ = read_yolo_labels(
            outputs_root / "clean" / "detections" / f"{name}.txt", with_conf=True
        )
        edge = cv2.imread(
            str(outputs_root / "clean" / "edges" / f"{name}.png"), cv2.IMREAD_GRAYSCALE
        )

        gt_overlay = _draw_boxes(img_rgb, gt_boxes, gt_classes, color=(0, 200, 0))
        pred_overlay = _draw_boxes(img_rgb, pred_boxes, pred_classes, color=(255, 80, 0))

        for col, (title, panel) in enumerate(
            [
                ("Clean", img_rgb),
                (f"{name} — GT", gt_overlay),
                (f"{name} — YOLOv8s", pred_overlay),
                (f"{name} — HED edges", edge),
            ]
        ):
            ax = axes[row, col]
            ax.imshow(panel, cmap=("gray" if col == 3 else None))
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)

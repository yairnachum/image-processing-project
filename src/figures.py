"""Matplotlib figure builders for Week 6+."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe for CI / headless runs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    for x, v in zip(df["class_name"], df[value_col]):
        ax.text(x, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    mean_val = float(df[value_col].mean())
    ax.axhline(mean_val, color="red", linestyle="--", linewidth=1, label=f"mean = {mean_val:.3f}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)

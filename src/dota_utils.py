"""DOTA v1.0 annotation utilities and subset sampler."""

import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle",
    "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool",
]

CLASS_COLORS = {
    cls: tuple(int(c) for c in color)
    for cls, color in zip(
        DOTA_CLASSES,
        [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
            (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
            (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
            (52, 69, 147), (100, 115, 255), (0, 24, 236),
        ],
    )
}


@dataclass
class Annotation:
    points: np.ndarray  # shape (4, 2), float32, pixel coords
    category: str
    difficult: int = 0


@dataclass
class Sample:
    image_path: Path
    annotations: List[Annotation] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.image_path.stem


def parse_dota_label(label_path: Path) -> List[Annotation]:
    """Parse a DOTA annotation txt file into a list of Annotation objects."""
    annotations = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("imagesource") or line.startswith("gsd"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            coords = list(map(float, parts[:8]))
            category = parts[8].lower()
            difficult = int(parts[9]) if len(parts) > 9 else 0
            points = np.array(coords, dtype=np.float32).reshape(4, 2)
            annotations.append(Annotation(points=points, category=category, difficult=difficult))
    return annotations


def load_dota_split(
    images_dir: Path,
    labels_dir: Path,
) -> List[Sample]:
    """Load all images that have a matching annotation file.

    Searches images_dir recursively (handles nested sub-folders from Drive downloads).
    labels_dir is also searched recursively for the matching .txt file.
    """
    # Build a flat map: stem -> label path
    label_map = {p.stem: p for p in labels_dir.rglob("*.txt")}

    samples = []
    for img_path in sorted(images_dir.rglob("*.png")) + sorted(images_dir.rglob("*.jpg")):
        lbl_path = label_map.get(img_path.stem)
        if lbl_path is not None:
            annotations = parse_dota_label(lbl_path)
            samples.append(Sample(image_path=img_path, annotations=annotations))
    return samples


def select_subset(samples: List[Sample], n: int = 200, seed: int = 7) -> List[Sample]:
    """Deterministically select n samples."""
    rng = random.Random(seed)
    pool = list(samples)
    rng.shuffle(pool)
    return pool[:n]


def train_test_split(
    samples: List[Sample], n_train: int = 160, seed: int = 7
) -> Tuple[List[Sample], List[Sample]]:
    """Split into train / test using the same seed."""
    rng = random.Random(seed)
    pool = list(samples)
    rng.shuffle(pool)
    return pool[:n_train], pool[n_train:]


def draw_annotations(
    image: np.ndarray,
    annotations: List[Annotation],
    thickness: int = 2,
    font_scale: float = 0.45,
) -> np.ndarray:
    """Draw oriented bounding boxes and class labels on a copy of the image."""
    out = image.copy()
    for ann in annotations:
        if ann.category not in CLASS_COLORS:
            color = (200, 200, 200)
        else:
            color = CLASS_COLORS[ann.category]
        pts = ann.points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
        x, y = int(ann.points[0, 0]), int(ann.points[0, 1])
        cv2.putText(
            out, ann.category, (x, max(y - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
        )
    return out


def make_sample_grid(
    samples: List[Sample],
    n_cols: int = 4,
    tile_size: int = 512,
    max_tiles: int = 16,
) -> np.ndarray:
    """Return an annotated image grid (H×W×3, uint8, RGB)."""
    chosen = samples[:max_tiles]
    n_rows = (len(chosen) + n_cols - 1) // n_cols
    grid_h = n_rows * tile_size
    grid_w = n_cols * tile_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, sample in enumerate(chosen):
        row, col = divmod(idx, n_cols)
        img_bgr = cv2.imread(str(sample.image_path))
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, (tile_size, tile_size))
        # Scale annotations to resized tile
        orig_h, orig_w = cv2.imread(str(sample.image_path)).shape[:2]  # re-read shape
        sx, sy = tile_size / orig_w, tile_size / orig_h
        scaled_anns = []
        for ann in sample.annotations:
            scaled_pts = ann.points * np.array([sx, sy], dtype=np.float32)
            scaled_anns.append(Annotation(points=scaled_pts, category=ann.category, difficult=ann.difficult))
        img_bgr = draw_annotations(img_bgr, scaled_anns)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        r0, c0 = row * tile_size, col * tile_size
        grid[r0:r0 + tile_size, c0:c0 + tile_size] = img_rgb

    return grid

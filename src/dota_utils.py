"""DOTA v1.0 annotation utilities, subset sampler, and tile/YOLO export."""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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
        orig_h, orig_w = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, (tile_size, tile_size))
        sx, sy = tile_size / orig_w, tile_size / orig_h
        scaled_anns = [
            Annotation(
                points=ann.points * np.array([sx, sy], dtype=np.float32),
                category=ann.category,
                difficult=ann.difficult,
            )
            for ann in sample.annotations
        ]
        img_resized = draw_annotations(img_resized, scaled_anns)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        r0, c0 = row * tile_size, col * tile_size
        grid[r0:r0 + tile_size, c0:c0 + tile_size] = img_rgb

    return grid


# ----- Tiling, AABB conversion, YOLO label export, manifest -----

def aabb_from_obb(points: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a 4-point OBB to an axis-aligned bbox (xmin, ymin, xmax, ymax)."""
    xmin = float(points[:, 0].min())
    ymin = float(points[:, 1].min())
    xmax = float(points[:, 0].max())
    ymax = float(points[:, 1].max())
    return xmin, ymin, xmax, ymax


def crop_to_tile(
    sample: Sample,
    tile_size: int = 1024,
) -> Tuple[np.ndarray, List[Annotation]]:
    """Crop one `tile_size × tile_size` tile centered on the annotation centroid.

    Source images smaller than `tile_size` are zero-padded to `tile_size` first.
    Annotations with no point inside the tile are dropped; partially-outside
    annotations are clipped to the tile boundary.

    Returns (BGR tile, translated annotations).
    """
    img = cv2.imread(str(sample.image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {sample.image_path}")
    h, w = img.shape[:2]

    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h or pad_w:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        h, w = img.shape[:2]

    if sample.annotations:
        all_pts = np.concatenate([a.points for a in sample.annotations], axis=0)
        cx, cy = all_pts.mean(axis=0).astype(int)
    else:
        cx, cy = w // 2, h // 2

    half = tile_size // 2
    x0 = int(max(0, min(w - tile_size, cx - half)))
    y0 = int(max(0, min(h - tile_size, cy - half)))
    tile = img[y0:y0 + tile_size, x0:x0 + tile_size]

    new_anns = []
    for ann in sample.annotations:
        new_pts = ann.points - np.array([x0, y0], dtype=np.float32)
        inside = (
            (new_pts[:, 0] >= 0) & (new_pts[:, 0] < tile_size) &
            (new_pts[:, 1] >= 0) & (new_pts[:, 1] < tile_size)
        )
        if inside.any():
            new_pts[:, 0] = np.clip(new_pts[:, 0], 0, tile_size - 1)
            new_pts[:, 1] = np.clip(new_pts[:, 1], 0, tile_size - 1)
            new_anns.append(
                Annotation(points=new_pts, category=ann.category, difficult=ann.difficult)
            )
    return tile, new_anns


def write_yolo_label(
    annotations: List[Annotation],
    out_path: Path,
    img_w: int,
    img_h: int,
) -> None:
    """Write YOLO labels: `class cx cy w h` (normalized AABB)."""
    lines = []
    for ann in annotations:
        if ann.category not in DOTA_CLASSES:
            continue
        cls = DOTA_CLASSES.index(ann.category)
        xmin, ymin, xmax, ymax = aabb_from_obb(ann.points)
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    out_path.write_text("\n".join(lines))


def write_yolo_obb_label(
    annotations: List[Annotation],
    out_path: Path,
    img_w: int,
    img_h: int,
) -> None:
    """Write DOTA-OBB YOLO labels: `cls x1 y1 x2 y2 x3 y3 x4 y4` (normalized)."""
    lines = []
    for ann in annotations:
        if ann.category not in DOTA_CLASSES:
            continue
        cls = DOTA_CLASSES.index(ann.category)
        pts = np.asarray(ann.points, dtype=np.float32).reshape(-1, 2)
        norm = []
        for x, y in pts:
            nx = min(max(x / img_w, 0.0), 1.0)
            ny = min(max(y / img_h, 0.0), 1.0)
            norm.extend([f"{nx:.6f}", f"{ny:.6f}"])
        lines.append(f"{cls} " + " ".join(norm))
    out_path.write_text("\n".join(lines))


def materialize_split(
    samples: List[Sample],
    out_root: Path,
    split: str,
    tile_size: int = 1024,
) -> List[Sample]:
    """Tile each sample, save image + YOLO label, return updated Samples."""
    img_dir = out_root / split / "images"
    lbl_dir = out_root / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    out_samples = []
    for s in samples:
        tile, anns = crop_to_tile(s, tile_size=tile_size)
        img_path = img_dir / f"{s.name}.png"
        lbl_path = lbl_dir / f"{s.name}.txt"
        cv2.imwrite(str(img_path), tile)
        write_yolo_label(anns, lbl_path, tile_size, tile_size)
        out_samples.append(Sample(image_path=img_path, annotations=anns))
    return out_samples


def write_dota_yaml(out_path: Path, data_root: Path) -> None:
    """Write the Ultralytics YOLO data config pointing at `data_root` (clean split)."""
    lines = [
        "# DOTA v1.0 — clean tiles (1024x1024)",
        f"path: {data_root.resolve()}",
        "train: train/images",
        "val: test/images",
        "",
        "names:",
        *(f"  {i}: {c}" for i, c in enumerate(DOTA_CLASSES)),
    ]
    out_path.write_text("\n".join(lines) + "\n")


def write_manifest(
    splits: Dict[str, List[Sample]],
    out_path: Path,
    seed: int = 7,
    tile_size: int = 1024,
) -> None:
    """Freeze the subset + train/test split as a small JSON manifest."""
    manifest = {
        "seed": seed,
        "tile_size": tile_size,
        "classes": DOTA_CLASSES,
        "splits": {name: [s.name for s in items] for name, items in splits.items()},
    }
    out_path.write_text(json.dumps(manifest, indent=2))

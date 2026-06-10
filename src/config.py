"""Project-wide paths, constants, and hyperparameters."""

from pathlib import Path

from src.dota_utils import DOTA_CLASSES  # re-export for downstream imports

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT    = PROJECT_ROOT / "data"
RAW_ROOT     = DATA_ROOT / "raw"
CLEAN_ROOT   = DATA_ROOT / "clean"
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

# --- Reproducibility ---
SEED = 7

# --- Subset / tiling ---
N_SUBSET  = 200
N_TRAIN   = 160
TILE_SIZE = 1024

# --- YOLOv8 ---
# We use the DOTA-OBB-pretrained variant (Ultralytics) so the baseline
# detects real DOTA classes out of the box. The Detector wrapper converts
# the OBB output to AABB (r.obb.xyxy) so the rest of the pipeline (YOLO
# AABB labels, IoU, mAP) stays unchanged. Class IDs 0–14 are identical to
# our DOTA_CLASSES list.
YOLO_WEIGHTS = "yolov8s-obb.pt"   # Ultralytics auto-downloads (DOTA-v1.0 pretrained)
YOLO_CONF    = 0.25
YOLO_IOU_NMS = 0.45

# --- HED (via controlnet_aux) ---
HED_REPO = "lllyasviel/Annotators"

# --- ORB ---
ORB_N_FEATURES = 1000
ORB_N_LEVELS   = 8

# Distortion sweep levels (W7+). Centralized here so the apply_distortions
# CLI and the W8 eval driver share one source of truth.
HAZE_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
JPEG_LEVELS = [1, 3, 5, 10, 20, 40]
NOISE_LEVELS = [5, 10, 15, 25, 35, 50]

# Week 10 — fine-tuning. yolov8s-obb specialists trained on distorted tiles.
FINETUNE_ROOT        = DATA_ROOT / "finetune"
FINETUNE_WEIGHTS_DIR = PROJECT_ROOT / "weights"
FINETUNE_N_TRAIN     = 128   # of the 160 train tiles; remaining 32 → val
FINETUNE_EPOCHS      = 50
FINETUNE_BATCH       = 4
FINETUNE_IMGSZ       = 1024

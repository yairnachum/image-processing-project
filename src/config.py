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
YOLO_WEIGHTS = "yolov8s.pt"   # Ultralytics auto-downloads
YOLO_CONF    = 0.25
YOLO_IOU_NMS = 0.45

# --- HED (via controlnet_aux) ---
HED_REPO = "lllyasviel/Annotators"

# --- ORB ---
ORB_N_FEATURES = 1000
ORB_N_LEVELS   = 8

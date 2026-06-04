"""YOLOv8 detector wrapper."""

from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

from src import config


class Detector:
    """Thin wrapper around an Ultralytics YOLO model.

    Always operates on RGB uint8 numpy arrays (HxWx3).
    """

    def __init__(self, weights: Optional[str] = None):
        self.weights = weights or config.YOLO_WEIGHTS
        self.model = YOLO(self.weights)

    def predict(self, image_rgb: np.ndarray) -> dict:
        r = self.model.predict(
            image_rgb,
            conf=config.YOLO_CONF,
            iou=config.YOLO_IOU_NMS,
            verbose=False,
        )[0]

        if r.boxes is None or len(r.boxes) == 0:
            return {
                "boxes_xyxy": np.zeros((0, 4), dtype=np.float32),
                "classes":    np.zeros((0,),  dtype=np.int32),
                "scores":     np.zeros((0,),  dtype=np.float32),
            }

        return {
            "boxes_xyxy": r.boxes.xyxy.cpu().numpy().astype(np.float32),
            "classes":    r.boxes.cls.cpu().numpy().astype(np.int32),
            "scores":     r.boxes.conf.cpu().numpy().astype(np.float32),
        }

    def train(self, data_yaml: Path, **kwargs) -> Path:
        """Fine-tune entry point used in Week 10. Not called from Week 5."""
        self.model.train(data=str(data_yaml), **kwargs)
        return Path(self.model.trainer.best)

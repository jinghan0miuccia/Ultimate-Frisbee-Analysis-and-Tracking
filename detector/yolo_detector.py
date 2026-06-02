from __future__ import annotations

from typing import Iterable

from ultralytics import YOLO

from utils.config import AppConfig


class YoloDetector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model = YOLO(config.model.path)

    def predict(self, frame) -> Iterable:
        return self.model.predict(
            source=frame,
            conf=self.config.model.confidence,
            iou=self.config.model.iou,
            imgsz=self.config.model.image_size,
            device=self.config.model.device,
            verbose=False,
        )

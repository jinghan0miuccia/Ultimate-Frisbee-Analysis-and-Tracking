from __future__ import annotations

from typing import Iterable

from ultralytics import YOLO

from utils.config import AppConfig
from utils.device import resolve_device


class YoloDetector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model = YOLO(config.model.path)
        self.device = resolve_device(config.model.device)

    def predict(self, frame) -> Iterable:
        return self.model.predict(
            source=frame,
            conf=self.config.model.confidence,
            iou=self.config.model.iou,
            imgsz=self.config.model.image_size,
            device=self.device,
            verbose=False,
        )

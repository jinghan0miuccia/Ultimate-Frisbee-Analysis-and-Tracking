from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from utils.logging import get_logger


LOGGER = get_logger(__name__)


class DepthEstimator:
    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = 0 if device and device.lower() in ("cuda", "gpu") and torch.cuda.is_available() else -1
        self.pipe: Any | None = None
        self.depth_map: np.ndarray | None = None

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        self._load_model()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pipe(Image.fromarray(rgb))
        depth = result["depth"]
        if isinstance(depth, Image.Image):
            depth_np = np.asarray(depth).astype(np.float32)
        else:
            depth_np = np.asarray(depth, dtype=np.float32)
        if depth_np.shape[:2] != frame.shape[:2]:
            depth_np = cv2.resize(depth_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        depth_min = float(depth_np.min())
        depth_max = float(depth_np.max())
        if depth_max > depth_min:
            depth_np = (depth_np - depth_min) / (depth_max - depth_min)
        self.depth_map = depth_np.astype(np.float32)
        return self.depth_map

    def get_depth(self, x: float, y: float) -> float:
        if self.depth_map is None:
            raise RuntimeError("Depth map is not available. Call estimate(frame) first.")
        height, width = self.depth_map.shape[:2]
        ix = min(max(int(round(x)), 0), width - 1)
        iy = min(max(int(round(y)), 0), height - 1)
        return float(self.depth_map[iy, ix])

    def _load_model(self) -> None:
        if self.pipe is not None:
            return
        LOGGER.info("Loading Depth Anything V2 model: %s", self.model_name)
        self.pipe = pipeline(task="depth-estimation", model=self.model_name, device=self.device)

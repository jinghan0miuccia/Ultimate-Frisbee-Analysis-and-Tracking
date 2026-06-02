from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from utils.logging import get_logger


LOGGER = get_logger(__name__)


class CoTrackerEstimator:
    def __init__(self, grid_size: int = 10, resize_width: int = 384, device: str | None = None) -> None:
        self.grid_size = grid_size
        self.resize_width = resize_width
        self.device = self._resolve_device(device)
        self.model: Any | None = None

    def estimate(self, frames: list[np.ndarray]) -> list[dict[str, float | int]]:
        if not frames:
            return []
        self._load_model()
        resized = [self._resize(frame) for frame in frames]
        video = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in resized], axis=0)
        tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(self.device)
        with torch.no_grad():
            tracks, visibility = self.model(tensor, grid_size=self.grid_size)
        tracks_np = tracks[0].detach().cpu().numpy()
        vis_np = visibility[0].detach().cpu().numpy().astype(bool)
        motion: list[dict[str, float | int]] = [{"frame": 1, "dx": 0.0, "dy": 0.0, "rotation": 0.0}]
        for idx in range(1, len(frames)):
            prev = tracks_np[idx - 1]
            curr = tracks_np[idx]
            visible = vis_np[idx - 1] & vis_np[idx]
            if visible.sum() < 4:
                motion.append({"frame": idx + 1, "dx": 0.0, "dy": 0.0, "rotation": 0.0})
                continue
            prev_v = prev[visible]
            curr_v = curr[visible]
            delta = curr_v - prev_v
            dx = float(np.median(delta[:, 0]))
            dy = float(np.median(delta[:, 1]))
            rotation = self._estimate_rotation(prev_v, curr_v)
            motion.append({"frame": idx + 1, "dx": dx, "dy": dy, "rotation": rotation})
        return motion

    def estimate_video(
        self,
        source: str,
        chunk_size: int = 120,
        start_sec: float = 0.0,
        max_frames: int | None = None,
    ) -> list[dict[str, float | int]]:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source for CoTracker: {source}")
        if start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
        all_motion: list[dict[str, float | int]] = []
        frame_offset = 0
        try:
            while True:
                frames: list[np.ndarray] = []
                while len(frames) < chunk_size:
                    if max_frames is not None and frame_offset + len(frames) >= max_frames:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                if not frames:
                    break
                chunk_motion = self.estimate(frames)
                for item in chunk_motion:
                    frame_id = frame_offset + int(item["frame"])
                    if frame_id == 1:
                        all_motion.append(item)
                    else:
                        all_motion.append(
                            {
                                "frame": frame_id,
                                "dx": float(item["dx"]),
                                "dy": float(item["dy"]),
                                "rotation": float(item["rotation"]),
                            }
                        )
                frame_offset += len(frames)
                if max_frames is not None and frame_offset >= max_frames:
                    break
        finally:
            cap.release()
        return all_motion

    def save(self, motion: list[dict[str, float | int]], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(motion, indent=2), encoding="utf-8")

    def _load_model(self) -> None:
        if self.model is not None:
            return
        LOGGER.info("Loading official CoTracker model via torch.hub")
        self.model = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline",
            trust_repo=True,
            skip_validation=True,
        ).to(self.device).eval()

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.resize_width:
            return frame
        scale = self.resize_width / float(width)
        return cv2.resize(frame, (self.resize_width, int(height * scale)), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _estimate_rotation(prev: np.ndarray, curr: np.ndarray) -> float:
        transform, _ = cv2.estimateAffinePartial2D(prev, curr, method=cv2.RANSAC)
        if transform is None:
            return 0.0
        return float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device and device.lower() in ("cuda", "gpu") and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

from __future__ import annotations

from typing import Dict, List

import cv2

from utils.config import VisualizationConfig
from utils.types import CenterPoint, TrackObject


class Visualizer:
    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config

    def draw(
        self,
        frame,
        objects: List[TrackObject],
        trajectories: Dict[int, List[CenterPoint]],
        fps: float,
        mode: str,
    ):
        output = frame.copy()
        for obj in objects:
            color = self._color(obj.track_id if obj.track_id >= 0 else hash(obj.class_name))
            x1, y1, x2, y2 = obj.bbox.as_int_tuple()
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.config.box_thickness)
            label_id = f"ID {obj.track_id} " if obj.track_id >= 0 else ""
            label = f"{label_id}{obj.class_name} {obj.confidence:.2f}"
            self._draw_label(output, label, x1, y1, color)
            if obj.track_id >= 0:
                self._draw_trajectory(output, trajectories.get(obj.track_id, []), color)

        status = f"{mode.upper()} | FPS {fps:.1f} | Tracks {len(objects)}"
        cv2.putText(output, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        return output

    def _draw_label(self, frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, self.config.font_scale, thickness)
        top = max(y - th - 8, 0)
        cv2.rectangle(frame, (x, top), (x + tw + 6, top + th + 6), color, -1)
        cv2.putText(frame, text, (x + 3, top + th + 2), font, self.config.font_scale, (0, 0, 0), thickness)

    def _draw_trajectory(self, frame, points: List[CenterPoint], color: tuple[int, int, int]) -> None:
        if len(points) < 2:
            return
        pts = [(int(p.x), int(p.y)) for p in points]
        for start, end in zip(pts[:-1], pts[1:]):
            cv2.line(frame, start, end, color, self.config.trajectory_thickness)

    @staticmethod
    def _color(seed: int) -> tuple[int, int, int]:
        seed = abs(seed)
        return (
            60 + (seed * 37) % 196,
            60 + (seed * 17) % 196,
            60 + (seed * 29) % 196,
        )

from __future__ import annotations

import cv2
import numpy as np

from scene.scene_coordinate import ScenePoint


class SceneRenderer:
    def __init__(self, width: int = 1000, height: int = 1000) -> None:
        self.width = width
        self.height = height
        self.origin = (width // 2, height // 2)

    def render(self, histories: dict[int, list[ScenePoint]], frame_id: int) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        active = 0
        for track_id, points in histories.items():
            if not points:
                continue
            active += 1
            color = self._color(track_id)
            pixel_points = [self._to_pixel(point) for point in points]
            for start, end in zip(pixel_points[:-1], pixel_points[1:]):
                cv2.line(canvas, start, end, color, 2)
            current = pixel_points[-1]
            cv2.circle(canvas, current, 6, color, -1)
            cv2.putText(canvas, str(track_id), (current[0] + 8, current[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(canvas, f"Frame {frame_id}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
        cv2.putText(canvas, f"Targets {active}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
        return canvas

    def _to_pixel(self, point: ScenePoint) -> tuple[int, int]:
        x = int(round(self.origin[0] + point.x))
        y = int(round(self.origin[1] + point.y))
        return min(max(x, 0), self.width - 1), min(max(y, 0), self.height - 1)

    @staticmethod
    def _color(seed: int) -> tuple[int, int, int]:
        seed = abs(seed)
        return 40 + (seed * 53) % 190, 40 + (seed * 97) % 190, 40 + (seed * 29) % 190

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CenterPoint:
    x: float
    y: float


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def center(self) -> CenterPoint:
        return CenterPoint(x=(self.x1 + self.x2) / 2.0, y=(self.y1 + self.y2) / 2.0)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass(frozen=True)
class TrackObject:
    track_id: int
    class_name: str
    confidence: float
    bbox: BBox
    center: CenterPoint
    timestamp: float
    frame_id: int

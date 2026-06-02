from __future__ import annotations

from dataclasses import dataclass

from utils.types import TrackObject


@dataclass(frozen=True)
class ScenePoint:
    x: float
    y: float


@dataclass(frozen=True)
class CameraPose:
    frame: int
    camera_x: float
    camera_y: float
    camera_rotation: float


class SceneCoordinateSystem:
    def __init__(self, frame_width: int, frame_height: int, scale: float = 180.0) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scale = scale
        self.norm = float(max(frame_width, frame_height))

    def project(self, track: TrackObject, depth: float, pose: CameraPose) -> ScenePoint:
        z = 0.5 + depth
        local_x = ((track.center.x - self.frame_width / 2.0) / self.norm) * self.scale * z
        local_y = ((track.center.y - self.frame_height / 2.0) / self.norm) * self.scale * z
        return ScenePoint(x=pose.camera_x + local_x, y=pose.camera_y + local_y)


class CameraPoseAccumulator:
    def __init__(self) -> None:
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_rotation = 0.0

    def update(self, frame_id: int, dx: float, dy: float, rotation: float) -> CameraPose:
        self.camera_x += dx
        self.camera_y += dy
        self.camera_rotation += rotation
        return CameraPose(
            frame=frame_id,
            camera_x=self.camera_x,
            camera_y=self.camera_y,
            camera_rotation=self.camera_rotation,
        )

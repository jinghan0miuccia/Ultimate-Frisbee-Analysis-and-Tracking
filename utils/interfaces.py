from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from utils.types import BBox, CenterPoint, TrackObject


class IReID(ABC):
    @abstractmethod
    def extract(self, image: Any, bbox: BBox) -> list[float]:
        raise NotImplementedError


class ICameraPoseProvider(ABC):
    @abstractmethod
    def get_pose(self, frame_id: int) -> Any:
        raise NotImplementedError


class IWorldCoordinateConverter(ABC):
    @abstractmethod
    def image_to_world(self, point: CenterPoint, track: TrackObject) -> Any:
        raise NotImplementedError

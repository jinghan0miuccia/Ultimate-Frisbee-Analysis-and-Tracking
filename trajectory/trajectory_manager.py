from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

from utils.types import CenterPoint, TrackObject


class TrajectoryManager:
    def __init__(self, max_history: int = 500) -> None:
        self.max_history = max_history
        self._history: Dict[int, Deque[CenterPoint]] = defaultdict(lambda: deque(maxlen=max_history))

    def update(self, obj: TrackObject) -> None:
        self._history[obj.track_id].append(obj.center)

    def query(self, track_id: int) -> List[CenterPoint]:
        return list(self._history.get(track_id, []))

    def histories(self) -> Dict[int, List[CenterPoint]]:
        return {track_id: list(points) for track_id, points in self._history.items()}

    def clear(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._history.clear()
        else:
            self._history.pop(track_id, None)

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from scene.scene_coordinate import ScenePoint


class TrajectoryDatabase:
    def __init__(self) -> None:
        self._tracks: dict[int, dict] = {}
        self._history: dict[int, list[ScenePoint]] = defaultdict(list)

    def update(self, track_id: int, class_name: str, point: ScenePoint, frame_id: int, depth: float) -> None:
        self._tracks[track_id] = {"track_id": track_id, "class": class_name}
        self._history[track_id].append(point)

    def query(self, track_id: int) -> list[ScenePoint]:
        return list(self._history.get(track_id, []))

    def all_tracks(self) -> dict[int, list[ScenePoint]]:
        return {track_id: list(points) for track_id, points in self._history.items()}

    def export(self, json_path: str | Path, csv_path: str | Path) -> None:
        json_records = []
        for track_id, points in sorted(self._history.items()):
            meta = self._tracks[track_id]
            json_records.append(
                {
                    "track_id": track_id,
                    "class": meta["class"],
                    "trajectory": [[point.x, point.y] for point in points],
                }
            )
        json_file = Path(json_path)
        csv_file = Path(csv_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        json_file.write_text(json.dumps(json_records, indent=2), encoding="utf-8")
        with csv_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["track_id", "class", "step", "scene_x", "scene_y"])
            writer.writeheader()
            for record in json_records:
                for step, (x, y) in enumerate(record["trajectory"], start=1):
                    writer.writerow(
                        {
                            "track_id": record["track_id"],
                            "class": record["class"],
                            "step": step,
                            "scene_x": x,
                            "scene_y": y,
                        }
                    )

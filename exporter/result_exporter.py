from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from utils.config import ExportConfig
from utils.logging import get_logger
from utils.types import TrackObject


LOGGER = get_logger(__name__)


class ResultExporter:
    def __init__(self, config: ExportConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)

    def export(self, tracks: Iterable[TrackObject]) -> None:
        if not self.config.enabled:
            return
        records = [self._record(track) for track in tracks]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(records)
        self._write_csv(records)
        LOGGER.info("Exported %d track records to %s", len(records), self.output_dir)

    def _write_json(self, records: list[dict]) -> None:
        path = self.output_dir / self.config.json_name
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def _write_csv(self, records: list[dict]) -> None:
        path = self.output_dir / self.config.csv_name
        fields = ["frame_id", "track_id", "class", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y", "timestamp"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def _record(track: TrackObject) -> dict:
        return {
            "frame_id": track.frame_id,
            "track_id": track.track_id,
            "class": track.class_name,
            "confidence": track.confidence,
            "x1": track.bbox.x1,
            "y1": track.bbox.y1,
            "x2": track.bbox.x2,
            "y2": track.bbox.y2,
            "center_x": track.center.x,
            "center_y": track.center.y,
            "timestamp": track.timestamp,
        }

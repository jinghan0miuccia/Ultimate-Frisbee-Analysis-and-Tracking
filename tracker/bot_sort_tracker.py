from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import List

import cv2
import yaml
from ultralytics import YOLO

from detector.yolo_detector import YoloDetector
from trajectory.trajectory_manager import TrajectoryManager
from utils.config import AppConfig
from utils.logging import get_logger
from utils.types import BBox, TrackObject
from utils.video import create_video_writer, open_video_capture
from visualization.visualizer import Visualizer


LOGGER = get_logger(__name__)


class BotSortTracker:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model = YOLO(config.model.path)
        self.detector = YoloDetector(config)
        self.trajectory = TrajectoryManager(max_history=config.trajectory.max_history)
        self.visualizer = Visualizer(config.visualization)
        self.class_ids = self._resolve_class_ids()
        self.tracker_config = self._prepare_tracker_config()

    def run_detection(self, source: str, display: bool = True, output_video: str | None = None) -> None:
        cap = open_video_capture(source)
        writer = create_video_writer(cap, output_video, self.config.video_output.codec) if output_video else None
        frame_id = 0
        last_time = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1
                results = list(self.detector.predict(frame))
                objects = self._detections_to_objects(results, frame_id)
                now = time.perf_counter()
                fps = 1.0 / max(now - last_time, 1e-6)
                last_time = now
                annotated = self.visualizer.draw(frame, objects, {}, fps=fps, mode="detect")
                if writer is not None:
                    writer.write(annotated)
                if display and not self._show(annotated):
                    break
        finally:
            if writer is not None:
                writer.release()
                LOGGER.info("Annotated video saved to %s", output_video)
            cap.release()
            cv2.destroyAllWindows()

    def run_tracking(self, source: str, display: bool = True, output_video: str | None = None) -> List[TrackObject]:
        cap = open_video_capture(source)
        writer = create_video_writer(cap, output_video, self.config.video_output.codec) if output_video else None
        frame_id = 0
        all_tracks: List[TrackObject] = []
        last_time = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1
                results = self.model.track(
                    source=frame,
                    stream=False,
                    persist=self.config.tracker.persist,
                    tracker=self.tracker_config,
                    conf=self.config.model.confidence,
                    iou=self.config.model.iou,
                    imgsz=self.config.model.image_size,
                    device=self.config.model.device,
                    classes=self.class_ids,
                    verbose=False,
                )
                objects = self._tracks_to_objects(results, frame_id)
                for obj in objects:
                    self.trajectory.update(obj)
                all_tracks.extend(objects)
                now = time.perf_counter()
                fps = 1.0 / max(now - last_time, 1e-6)
                last_time = now
                annotated = self.visualizer.draw(
                    frame,
                    objects,
                    self.trajectory.histories(),
                    fps=fps,
                    mode="track",
                )
                if writer is not None:
                    writer.write(annotated)
                if display and not self._show(annotated):
                    break
        finally:
            if writer is not None:
                writer.release()
                LOGGER.info("Annotated video saved to %s", output_video)
            cap.release()
            cv2.destroyAllWindows()
        return all_tracks

    def _show(self, frame) -> bool:
        cv2.imshow(self.config.visualization.window_name, frame)
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)

    def _resolve_class_ids(self) -> List[int] | None:
        names = self.model.names
        wanted = set(self.config.classes.filter)
        if not wanted:
            return None
        return [int(idx) for idx, name in names.items() if name in wanted]

    def _prepare_tracker_config(self) -> str:
        if not self.config.tracker.parameters:
            return self.config.tracker.config_path
        output = dict(self.config.tracker.parameters)
        output["tracker_type"] = "botsort"
        tmp = tempfile.NamedTemporaryFile("w", suffix="_botsort.yaml", delete=False, encoding="utf-8")
        with tmp:
            yaml.safe_dump(output, tmp, sort_keys=False)
        LOGGER.info("Using BoT-SORT tracker config: %s", tmp.name)
        return tmp.name

    def _detections_to_objects(self, results, frame_id: int) -> List[TrackObject]:
        timestamp = time.time()
        objects: List[TrackObject] = []
        if not results:
            return objects
        result = results[0]
        if result.boxes is None:
            return objects
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            class_name = names.get(cls_id, str(cls_id))
            if self.config.classes.filter and class_name not in self.config.classes.filter:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
            objects.append(
                TrackObject(
                    track_id=-1,
                    class_name=class_name,
                    confidence=float(box.conf.item()),
                    bbox=bbox,
                    center=bbox.center(),
                    timestamp=timestamp,
                    frame_id=frame_id,
                )
            )
        return objects

    def _tracks_to_objects(self, results, frame_id: int) -> List[TrackObject]:
        timestamp = time.time()
        objects: List[TrackObject] = []
        if not results:
            return objects
        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return objects
        names = result.names
        for box in result.boxes:
            if box.id is None:
                continue
            cls_id = int(box.cls.item())
            class_name = names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
            objects.append(
                TrackObject(
                    track_id=int(box.id.item()),
                    class_name=class_name,
                    confidence=float(box.conf.item()),
                    bbox=bbox,
                    center=bbox.center(),
                    timestamp=timestamp,
                    frame_id=frame_id,
                )
            )
        return objects

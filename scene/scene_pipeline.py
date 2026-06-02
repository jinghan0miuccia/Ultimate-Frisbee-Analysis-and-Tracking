from __future__ import annotations

from pathlib import Path
from typing import List

import cv2

from depth.depth_estimator import DepthEstimator
from motion.cotracker_estimator import CoTrackerEstimator
from scene.scene_coordinate import CameraPoseAccumulator, SceneCoordinateSystem
from scene.scene_renderer import SceneRenderer
from scene.trajectory_database import TrajectoryDatabase
from tracker.bot_sort_tracker import BotSortTracker
from utils.config import AppConfig
from utils.logging import get_logger
from utils.types import TrackObject
from utils.video import create_video_writer, open_video_capture


LOGGER = get_logger(__name__)


class ScenePipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tracker = BotSortTracker(config)
        self.motion = CoTrackerEstimator(
            grid_size=config.scene.cotracker_grid_size,
            resize_width=config.scene.cotracker_resize_width,
            device=config.model.device,
        )
        self.depth = DepthEstimator(model_name=config.scene.depth_model, device=config.model.device)

    def run(self, source: str, start_sec: float = 0.0, max_frames: int | None = None) -> None:
        output_dir = Path(self.config.export.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        camera_motion = self.motion.estimate_video(
            source,
            self.config.scene.cotracker_chunk_size,
            start_sec=start_sec,
            max_frames=max_frames,
        )
        if not camera_motion:
            raise RuntimeError(f"No camera motion estimated from source: {source}")
        self.motion.save(camera_motion, output_dir / self.config.scene.camera_motion_json)

        cap = open_video_capture(source)
        if start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
        annotated_writer = create_video_writer(cap, str(output_dir / self.config.scene.annotated_name), self.config.video_output.codec)
        map_writer = cv2.VideoWriter(
            str(output_dir / self.config.scene.scene_map_name),
            cv2.VideoWriter_fourcc(*self.config.video_output.codec),
            cap.get(cv2.CAP_PROP_FPS) or 25.0,
            (self.config.scene.map_width, self.config.scene.map_height),
        )
        if not map_writer.isOpened():
            raise RuntimeError("Unable to open scene map writer")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        coord = SceneCoordinateSystem(frame_width, frame_height, self.config.scene.map_scale)
        pose_accumulator = CameraPoseAccumulator()
        db = TrajectoryDatabase()
        renderer = SceneRenderer(self.config.scene.map_width, self.config.scene.map_height)
        try:
            frame_id = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1
                if max_frames is not None and frame_id > max_frames:
                    break
                tracks, annotated = self.tracker.process_frame(frame, frame_id, mode="scene")
                depth_map = self.depth.estimate(frame)
                motion = camera_motion[min(frame_id - 1, len(camera_motion) - 1)]
                pose = pose_accumulator.update(
                    frame_id,
                    float(motion["dx"]),
                    float(motion["dy"]),
                    float(motion["rotation"]),
                )
                self._project_tracks(tracks, db, coord, pose)
                annotated_writer.write(annotated)
                map_writer.write(renderer.render(db.all_tracks(), frame_id))
                LOGGER.info("Scene frame %d: tracks=%d depth=%s", frame_id, len(tracks), depth_map.shape)
        finally:
            cap.release()
            annotated_writer.release()
            map_writer.release()
            cv2.destroyAllWindows()
        db.export(output_dir / self.config.scene.scene_tracks_json, output_dir / self.config.scene.scene_tracks_csv)
        LOGGER.info("Scene outputs written to %s", output_dir)

    def _project_tracks(self, tracks: List[TrackObject], db: TrajectoryDatabase, coord: SceneCoordinateSystem, pose) -> None:
        for track in tracks:
            depth = self.depth.get_depth(track.center.x, track.center.y)
            point = coord.project(track, depth, pose)
            db.update(track.track_id, track.class_name, point, track.frame_id, depth)

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    path: str = "models/yolo11x.pt"
    confidence: float = 0.25
    iou: float = 0.7
    image_size: int = 640
    device: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str = "track"
    log_level: str = "INFO"


@dataclass(frozen=True)
class TrackerConfig:
    config_path: str = "botsort.yaml"
    persist: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassesConfig:
    filter: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrajectoryConfig:
    max_history: int = 500


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool = True
    window_name: str = "YOLO11x BoT-SORT Tracking"
    box_thickness: int = 2
    trajectory_thickness: int = 2
    font_scale: float = 0.55


@dataclass(frozen=True)
class VideoOutputConfig:
    enabled: bool = False
    path: str | None = None
    codec: str = "mp4v"


@dataclass(frozen=True)
class ExportConfig:
    enabled: bool = True
    output_dir: str = "outputs"
    json_name: str = "tracks.json"
    csv_name: str = "tracks.csv"


@dataclass(frozen=True)
class SceneConfig:
    annotated_name: str = "annotated.mp4"
    scene_map_name: str = "scene_map.mp4"
    scene_tracks_json: str = "scene_tracks.json"
    scene_tracks_csv: str = "scene_tracks.csv"
    camera_motion_json: str = "camera_motion.json"
    map_width: int = 1000
    map_height: int = 1000
    map_scale: float = 180.0
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    cotracker_grid_size: int = 10
    cotracker_resize_width: int = 384
    cotracker_chunk_size: int = 120


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    classes: ClassesConfig = field(default_factory=ClassesConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    video_output: VideoOutputConfig = field(default_factory=VideoOutputConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)


def load_config(path: str | Path) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AppConfig(
        model=ModelConfig(**data.get("model", {})),
        runtime=RuntimeConfig(**data.get("runtime", {})),
        tracker=TrackerConfig(**data.get("tracker", {})),
        classes=ClassesConfig(**data.get("classes", {})),
        trajectory=TrajectoryConfig(**data.get("trajectory", {})),
        visualization=VisualizationConfig(**data.get("visualization", {})),
        video_output=VideoOutputConfig(**data.get("video_output", {})),
        export=ExportConfig(**data.get("export", {})),
        scene=SceneConfig(**data.get("scene", {})),
    )

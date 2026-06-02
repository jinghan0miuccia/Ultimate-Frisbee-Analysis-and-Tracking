from __future__ import annotations

import argparse
import sys
from pathlib import Path

from exporter.result_exporter import ResultExporter
from scene.scene_pipeline import ScenePipeline
from tracker.bot_sort_tracker import BotSortTracker
from utils.config import AppConfig, load_config
from utils.logging import configure_logging, get_logger
from utils.video import validate_video_source


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11x + BoT-SORT multi-object tracking")
    parser.add_argument("source", help="Video file, camera index, or RTSP URL")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--mode",
        choices=("env", "detect", "track", "scene"),
        default=None,
        help="Run environment check, detection, tracking, or scene reconstruction. Defaults to config runtime.mode.",
    )
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV preview window")
    parser.add_argument(
        "--save-video",
        default=None,
        help="Write the annotated output video to this path, for example outputs/annotated.mp4",
    )
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start processing from this timestamp in seconds")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    return parser.parse_args()


def run_environment_check(config: AppConfig, source: str) -> None:
    from ultralytics import YOLO
    from utils.device import resolve_device

    LOGGER.info("Loading model: %s", config.model.path)
    _ = YOLO(config.model.path)
    LOGGER.info("Validating device: %s", resolve_device(config.model.device))
    LOGGER.info("Validating video source: %s", source)
    validate_video_source(source)
    LOGGER.info("Validating official BoT-SORT tracker: %s", config.tracker.config_path)
    if not Path(config.tracker.config_path).exists() and config.tracker.config_path != "botsort.yaml":
        raise FileNotFoundError(f"Tracker config not found: {config.tracker.config_path}")
    print("Environment Ready")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.runtime.log_level)
    mode = args.mode or config.runtime.mode
    display = config.visualization.enabled and not args.no_display
    output_video = args.save_video or config.video_output.path

    try:
        run_environment_check(config, args.source)
        if mode == "env":
            return 0

        if mode == "scene":
            scene = ScenePipeline(config)
            scene.run(source=args.source, start_sec=args.start_sec, max_frames=args.max_frames)
            return 0

        exporter = ResultExporter(config.export)
        tracker = BotSortTracker(config)

        if mode == "detect":
            tracker.run_detection(source=args.source, display=display, output_video=output_video)
            return 0

        tracks = tracker.run_tracking(source=args.source, display=display, output_video=output_video)
        exporter.export(tracks)
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
        return 130
    except Exception as exc:
        LOGGER.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

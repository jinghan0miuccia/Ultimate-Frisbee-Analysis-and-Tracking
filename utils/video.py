from __future__ import annotations

from pathlib import Path

import cv2


def parse_source(source: str) -> int | str:
    if source.isdigit():
        return int(source)
    return source


def open_video_capture(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(parse_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    return cap


def validate_video_source(source: str) -> None:
    cap = open_video_capture(source)
    try:
        ok, _ = cap.read()
        if not ok:
            raise RuntimeError(f"Unable to read first frame from source: {source}")
    finally:
        cap.release()


def create_video_writer(cap: cv2.VideoCapture, output_path: str, codec: str = "mp4v") -> cv2.VideoWriter:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        raise RuntimeError("Unable to determine source frame size")
    if fps <= 0 or fps != fps:
        fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer: {output_path}")
    return writer

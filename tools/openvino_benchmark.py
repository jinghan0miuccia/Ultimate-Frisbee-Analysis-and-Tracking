from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from openvino import Core
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO11s OpenVINO on Intel devices")
    parser.add_argument("source", help="Video path, camera index, or RTSP URL")
    parser.add_argument("--model", default="models/yolo11s.pt", help="Source PyTorch model path")
    parser.add_argument("--openvino-model", default="models/yolo11s_openvino_model", help="OpenVINO model directory")
    parser.add_argument("--device", default="intel:gpu", help='Ultralytics device, for example "intel:gpu" or "cpu"')
    parser.add_argument("--frames", type=int, default=300, help="Maximum frames to benchmark")
    parser.add_argument("--mode", choices=("predict", "track"), default="track")
    parser.add_argument("--output", default="outputs/openvino_benchmark.mp4", help="Annotated output video path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def export_openvino_model(model_path: str, output_dir: str) -> Path:
    output = Path(output_dir)
    if output.exists() and any(output.glob("*.xml")):
        return output
    model = YOLO(model_path)
    exported = model.export(format="openvino")
    exported_path = Path(exported)
    if exported_path.resolve() != output.resolve():
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            exported_path.rename(output)
    return output


def create_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output video: {output_path}")
    return writer


def list_openvino_devices() -> None:
    core = Core()
    print("OpenVINO devices:", ", ".join(core.available_devices))


def main() -> int:
    args = parse_args()
    list_openvino_devices()
    ov_path = export_openvino_model(args.model, args.openvino_model)
    model = YOLO(str(ov_path), task="detect")

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")
    writer = create_writer(cap, args.output)

    frame_count = 0
    object_count = 0
    start = time.perf_counter()
    try:
        while frame_count < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            if args.mode == "track":
                results = model.track(
                    frame,
                    tracker="botsort.yaml",
                    persist=True,
                    device=args.device,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    verbose=False,
                )
            else:
                results = model.predict(
                    frame,
                    device=args.device,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    verbose=False,
                )
            result = results[0]
            object_count += 0 if result.boxes is None else len(result.boxes)
            writer.write(result.plot())
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"mode={args.mode}")
    print(f"device={args.device}")
    print(f"frames={frame_count}")
    print(f"objects={object_count}")
    print(f"elapsed_sec={elapsed:.2f}")
    print(f"fps={fps:.2f}")
    print(f"output={Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

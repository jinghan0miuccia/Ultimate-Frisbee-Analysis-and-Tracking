# Ultimate-Frisbee-Analysis-and-Tracking

Cooperate with CocoWang (Peter He). We want to create a model that can auto-track the players of Ultimate frisbee from the input video. Eventually, LLM will be interfaced to the project, providing practical use for the coach to sophisticate players performance.

Realtime multi-object tracking with the official Ultralytics YOLO11 API and BoT-SORT tracker.

## Stack

- YOLO11s model: `models/yolo11s.pt`
- Device: GPU by default with `model.device: cuda`
- Tracker: official Ultralytics BoT-SORT via `model.track(..., tracker="botsort.yaml", persist=True)`
- Inputs: video file, camera index, RTSP URL
- Outputs: OpenCV visualization, annotated MP4, JSON, CSV

## Install

```powershell
pip install -r requirements.txt
```

If `python` opens the Windows Store stub on Windows, use the real interpreter path or `py`.

For NVIDIA GPU inference, install a CUDA-enabled PyTorch build before running:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

Then verify:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

`config/config.yaml` uses `model.device: cuda` by default. If CUDA is unavailable, the program exits with a clear error instead of silently falling back to CPU.

## Run

Environment check:

```powershell
python main.py test.mp4 --mode env
```

Detection only:

```powershell
python main.py test.mp4 --mode detect
```

Tracking with visualization and export:

```powershell
python main.py test.mp4 --mode track
```

Write an annotated video:

```powershell
python main.py test.mp4 --mode track --save-video outputs/annotated.mp4
```

Camera:

```powershell
python main.py 0 --mode track
```

RTSP:

```powershell
python main.py rtsp://user:pass@host:554/stream --mode track
```

Headless export:

```powershell
python main.py test.mp4 --mode track --no-display
```

## Configuration

Edit `config/config.yaml` for model path, thresholds, tracker parameters, output path, and class filter.

Tracked default classes:

- person
- car
- bus
- truck
- motorcycle
- bicycle

## Architecture

```text
main.py
config/              YAML configuration
detector/            YOLO11 detection-only wrapper
tracker/             Official BoT-SORT tracking wrapper
trajectory/          Per-track trajectory history
visualization/       OpenCV drawing
exporter/            JSON/CSV export
models/              Runtime model files
outputs/             Exported tracking results
utils/               Config, logging, source validation, common types
```

Core data moves through `TrackObject`, not loose dictionaries. Future integration points are declared in `utils/interfaces.py` for ReID, camera pose, and world coordinate conversion.

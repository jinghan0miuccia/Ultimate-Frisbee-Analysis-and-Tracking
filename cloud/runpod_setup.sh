#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/jinghan0miuccia/Ultimate-Frisbee-Analysis-and-Tracking.git}"
WORKDIR_PATH="${WORKDIR_PATH:-/workspace/YOLOTrack}"

echo "[1/6] GPU check"
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

echo "[2/6] System packages"
apt-get update
apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0

echo "[3/6] Clone or update repo"
if [ -d "$WORKDIR_PATH/.git" ]; then
  git -C "$WORKDIR_PATH" pull --ff-only
else
  git clone "$REPO_URL" "$WORKDIR_PATH"
fi
cd "$WORKDIR_PATH"

echo "[4/6] Python dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[5/6] Warm model caches"
python - <<'PY'
from ultralytics import YOLO
YOLO("models/yolo11x.pt")
print("YOLO11x ready")
PY

echo "[6/6] Ready"
echo "Upload your video to $WORKDIR_PATH/data/input.mp4, then run:"
echo "python main.py data/input.mp4 --config config/config.gpu.yaml --mode scene --no-display"

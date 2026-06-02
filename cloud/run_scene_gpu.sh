#!/usr/bin/env bash
set -euo pipefail

VIDEO="${1:-data/input.mp4}"
START_SEC="${START_SEC:-0}"
MAX_FRAMES="${MAX_FRAMES:-}"

ARGS=( "$VIDEO" --config config/config.gpu.yaml --mode scene --no-display --start-sec "$START_SEC" )
if [ -n "$MAX_FRAMES" ]; then
  ARGS+=( --max-frames "$MAX_FRAMES" )
fi

python main.py "${ARGS[@]}"

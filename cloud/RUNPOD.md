# RunPod GPU Deployment

Recommended service: RunPod Pods with an official PyTorch template.

Why RunPod:

- Official PyTorch templates include CUDA PyTorch and SSH access.
- Pods can attach persistent storage for videos, model caches, and outputs.
- This workload is batch video processing, so hourly GPU Pods are a good fit.

## Recommended GPU

Start with one of:

- RTX 4090 24 GB
- RTX 3090 24 GB
- RTX A5000 24 GB
- A10 / A10G 24 GB

Avoid GPUs below 16 GB VRAM for full `YOLO11x + CoTracker + Depth Anything V2` scene mode.

## Pod Setup

1. Create a RunPod Pod.
2. Choose an official PyTorch CUDA template.
3. Use at least 50 GB container disk, or attach a persistent volume.
4. Connect with SSH or Web Terminal.
5. Run:

```bash
curl -fsSL https://raw.githubusercontent.com/jinghan0miuccia/Ultimate-Frisbee-Analysis-and-Tracking/main/cloud/runpod_setup.sh | bash
```

## Upload Video

Upload the input file to:

```text
/workspace/YOLOTrack/data/input.mp4
```

With `scp`:

```bash
scp FrisbeeTest.mp4 root@YOUR_RUNPOD_HOST:/workspace/YOLOTrack/data/input.mp4
```

Use RunPod full SSH over public IP for `scp`/SFTP. If the Pod only exposes proxied SSH, upload through JupyterLab or VSCode instead.

## Run Scene Mode

Full video:

```bash
cd /workspace/YOLOTrack
python main.py data/input.mp4 --config config/config.gpu.yaml --mode scene --no-display
```

Segment test:

```bash
cd /workspace/YOLOTrack
python main.py data/input.mp4 --config config/config.gpu.yaml --mode scene --no-display --start-sec 60 --max-frames 300
```

Or use the helper:

```bash
cd /workspace/YOLOTrack
START_SEC=60 MAX_FRAMES=300 bash cloud/run_scene_gpu.sh data/input.mp4
```

## Outputs

Files are written to:

```text
/workspace/YOLOTrack/outputs/
```

Expected files:

- `annotated.mp4`
- `scene_map.mp4`
- `scene_tracks.json`
- `scene_tracks.csv`
- `camera_motion.json`

Download them with:

```bash
scp root@YOUR_RUNPOD_HOST:/workspace/YOLOTrack/outputs/annotated.mp4 .
scp root@YOUR_RUNPOD_HOST:/workspace/YOLOTrack/outputs/scene_map.mp4 .
scp root@YOUR_RUNPOD_HOST:/workspace/YOLOTrack/outputs/scene_tracks.json .
scp root@YOUR_RUNPOD_HOST:/workspace/YOLOTrack/outputs/scene_tracks.csv .
scp root@YOUR_RUNPOD_HOST:/workspace/YOLOTrack/outputs/camera_motion.json .
```

## Docker Alternative

Build:

```bash
docker build -f Dockerfile.gpu -t yolotrack-gpu .
```

Run:

```bash
docker run --gpus all --rm -it \
  -v "$PWD/data:/workspace/YOLOTrack/data" \
  -v "$PWD/outputs:/workspace/YOLOTrack/outputs" \
  yolotrack-gpu \
  python main.py data/input.mp4 --config config/config.gpu.yaml --mode scene --no-display
```

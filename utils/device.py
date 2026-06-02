from __future__ import annotations


def resolve_device(device: str | None) -> str | int | None:
    if device is None:
        return None
    normalized = str(device).strip().lower()
    if normalized in ("", "none", "auto"):
        return None
    if normalized in ("cuda", "gpu"):
        _ensure_cuda()
        return 0
    if normalized.startswith("cuda:"):
        _ensure_cuda()
        return int(normalized.split(":", 1)[1])
    if normalized.isdigit():
        _ensure_cuda()
        return int(normalized)
    if normalized == "cpu":
        return "cpu"
    return device


def _ensure_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU requested, but the installed PyTorch build cannot use CUDA. "
            "Install a CUDA-enabled torch/torchvision build, or set model.device to cpu/null."
        )

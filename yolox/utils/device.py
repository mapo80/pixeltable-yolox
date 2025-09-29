"""Utility helpers for selecting the compute device (CUDA, MPS, CPU)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch

__all__ = [
    "YOLOX_DEVICE_ENV",
    "get_target_device_type",
    "get_target_device",
    "is_cuda_device",
    "supports_amp",
    "autocast_context",
]

YOLOX_DEVICE_ENV = "YOLOX_DEVICE"


def _normalize_choice(choice: str | None) -> str:
    if not choice:
        return "auto"
    choice = choice.strip().lower()
    if choice in {"gpu"}:
        return "cuda"
    if choice in {"cpu", "cuda", "mps", "auto"}:
        return choice
    raise ValueError(
        f"Unsupported device '{choice}'. Expected one of: auto, cuda, mps, cpu"
    )


def get_target_device_type() -> str:
    """Return the desired device type based on availability and env override."""

    choice = _normalize_choice(os.getenv(YOLOX_DEVICE_ENV))
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "YOLOX_DEVICE=cuda but CUDA is not available. "
                "Check your PyTorch build or unset YOLOX_DEVICE."
            )
        return "cuda"

    if choice == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError(
                "YOLOX_DEVICE=mps but MPS backend is not available. "
                "Make sure you're running on macOS with an Apple GPU and a recent PyTorch build."
            )
        return "mps"

    # explicit CPU request
    return "cpu"


def get_target_device() -> torch.device:
    return torch.device(get_target_device_type())


def is_cuda_device(device: torch.device | str | None = None) -> bool:
    if device is None:
        device = get_target_device()
    if isinstance(device, str):
        return device.startswith("cuda")
    return device.type == "cuda"


def supports_amp(device_type: str) -> bool:
    return device_type == "cuda"


@contextmanager
def autocast_context(device_type: str, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    if device_type == "cuda":
        with torch.cuda.amp.autocast():
            yield
        return

    # No autocast fallback for CPU/MPS yet.
    yield

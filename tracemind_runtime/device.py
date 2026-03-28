"""런타임 장치 선택 유틸리티."""

from __future__ import annotations

import importlib
import re
import warnings
from typing import Any

_CUDA_DEVICE_PATTERN = re.compile(r"^cuda(?::(?P<index>\d+))?$")
_SUPPORTED_DEVICE_HINT = "Supported values: auto, cpu, cuda, cuda:<index>, mps."


def resolve_runtime_device(device: str | None = "auto") -> str:
    """요청 문자열을 실제 런타임 장치 이름으로 정규화한다."""
    requested = _normalize_device(device)
    if requested == "auto":
        if _cuda_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if requested == "mps":
        if _mps_available():
            return "mps"
        raise RuntimeError("Requested runtime device 'mps' is not available.")

    cuda_match = _CUDA_DEVICE_PATTERN.fullmatch(requested)
    if cuda_match is not None:
        _require_cuda_device(cuda_match.group("index"))
        return requested

    raise ValueError(f"Unsupported runtime device selector '{device}'. {_SUPPORTED_DEVICE_HINT}")


def _normalize_device(device: str | None) -> str:
    if device is None:
        return "auto"

    normalized = device.strip().lower()
    if not normalized:
        return "auto"
    if normalized == "gpu":
        return "cuda"
    if normalized.startswith("gpu:"):
        return f"cuda:{normalized.split(':', maxsplit=1)[1]}"
    return normalized


def _load_torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _cuda_available() -> bool:
    torch = _load_torch()
    if torch is None or getattr(torch, "cuda", None) is None:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if not torch.cuda.is_available():
                return False
            device_count = getattr(torch.cuda, "device_count", None)
            if callable(device_count):
                return int(device_count()) > 0
            return True
        except Exception:
            return False


def _mps_available() -> bool:
    torch = _load_torch()
    if torch is None:
        return False

    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    if mps_backend is None:
        return False

    try:
        return bool(mps_backend.is_built() and mps_backend.is_available())
    except Exception:
        return False


def _require_cuda_device(index_text: str | None) -> None:
    if not _cuda_available():
        raise RuntimeError("Requested runtime device 'cuda' is not available.")

    if index_text is None:
        return

    torch = _load_torch()
    if torch is None or getattr(torch, "cuda", None) is None:
        raise RuntimeError("Requested runtime device 'cuda' is not available.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            device_count = int(torch.cuda.device_count())
        except Exception as exc:
            raise RuntimeError("Requested CUDA device index could not be validated.") from exc

    if int(index_text) >= device_count:
        raise RuntimeError(
            f"Requested runtime device 'cuda:{index_text}' is unavailable. "
            f"Detected CUDA device count: {device_count}."
        )

"""Runtime device selection unit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracemind_runtime.device import resolve_runtime_device


def _fake_torch(
    *,
    cuda_available: bool,
    cuda_device_count: int,
    mps_available: bool = False,
    mps_built: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            device_count=lambda: cuda_device_count,
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(
                is_built=lambda: mps_built,
                is_available=lambda: mps_available,
            )
        ),
    )


def test_resolve_runtime_device_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tracemind_runtime.device._load_torch",
        lambda: _fake_torch(cuda_available=True, cuda_device_count=2),
    )

    assert resolve_runtime_device("auto") == "cuda"


def test_resolve_runtime_device_falls_back_to_cpu_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tracemind_runtime.device._load_torch", lambda: None)

    assert resolve_runtime_device("auto") == "cpu"


def test_resolve_runtime_device_uses_mps_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tracemind_runtime.device._load_torch",
        lambda: _fake_torch(
            cuda_available=False,
            cuda_device_count=0,
            mps_available=True,
            mps_built=True,
        ),
    )

    assert resolve_runtime_device("auto") == "mps"


def test_resolve_runtime_device_rejects_missing_explicit_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tracemind_runtime.device._load_torch",
        lambda: _fake_torch(cuda_available=False, cuda_device_count=0),
    )

    with pytest.raises(RuntimeError, match="Requested runtime device 'cuda' is not available"):
        resolve_runtime_device("cuda")


def test_resolve_runtime_device_validates_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tracemind_runtime.device._load_torch",
        lambda: _fake_torch(cuda_available=True, cuda_device_count=1),
    )

    with pytest.raises(RuntimeError, match="cuda:1"):
        resolve_runtime_device("cuda:1")


def test_resolve_runtime_device_supports_gpu_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tracemind_runtime.device._load_torch",
        lambda: _fake_torch(cuda_available=True, cuda_device_count=1),
    )

    assert resolve_runtime_device("gpu") == "cuda"

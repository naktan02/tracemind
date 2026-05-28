"""PEFT encoder classifier training batch iteration helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader


def next_cycling_batch(
    *,
    loader: DataLoader[dict[str, Any]],
    iterator: Iterator[dict[str, Any]],
) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    """loader 끝에 도달하면 새 iterator로 이어서 다음 batch를 반환한다."""

    try:
        return next(iterator), iterator
    except StopIteration:
        refreshed_iterator = iter(loader)
        return next(refreshed_iterator), refreshed_iterator


def move_tensor_batch_to_device(
    *,
    batch: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    """tensor 값만 target device로 이동하고 non-tensor metadata는 보존한다."""

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(
                device,
                non_blocking=_is_cuda_device(device),
            )
        else:
            moved[key] = value
    return moved


def _is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda")

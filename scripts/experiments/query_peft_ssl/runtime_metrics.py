"""Query LoRA 학습 비용 지표 측정 helper."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import TypeVar

import torch

from shared.src.domain.services.classification_report import safe_divide

T = TypeVar("T")


def run_with_training_runtime_metrics(
    train_callable: Callable[[], T],
    *,
    device: str,
    training_example_count: int,
    parameter_counts: Mapping[str, int],
) -> tuple[T, dict[str, float | int | None]]:
    """학습 호출을 실행하고 run-level 비용 지표를 계산한다."""

    _synchronize_if_cuda(device)
    started_at = time.perf_counter()
    result = train_callable()
    _synchronize_if_cuda(device)
    train_seconds = time.perf_counter() - started_at

    trainable_parameters = int(parameter_counts.get("trainable", 0))
    total_parameters = int(parameter_counts.get("total", 0))
    metrics: dict[str, float | int | None] = {
        "train_seconds": round(train_seconds, 6),
        "training_example_count": int(training_example_count),
        "examples_per_second": round(
            safe_divide(float(training_example_count), train_seconds),
            6,
        ),
        "trainable_param_ratio": round(
            safe_divide(float(trainable_parameters), float(total_parameters)),
            8,
        ),
    }
    return result, metrics


def _synchronize_if_cuda(device: str) -> None:
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return
    torch.cuda.synchronize(torch.device(device))

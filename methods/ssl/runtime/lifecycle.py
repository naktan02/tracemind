"""Query SSL trainer lifecycle dispatch helpers."""

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch import device as TorchDevice

from methods.ssl.base import (
    QuerySslAlgorithm,
    QuerySslStepContext,
    QuerySslStepResult,
    TextBatchClassifier,
)


def configure_query_ssl_algorithm_training(
    algorithm: QuerySslAlgorithm,
    *,
    num_train_iter: int,
) -> None:
    """algorithm이 필요로 할 때만 전체 train iteration 수를 전달한다."""

    if num_train_iter <= 0:
        raise ValueError("num_train_iter must be positive.")
    configure_training = getattr(algorithm, "configure_training", None)
    if callable(configure_training):
        configure_training(num_train_iter=num_train_iter)


def configure_query_ssl_algorithm_batching(
    algorithm: QuerySslAlgorithm,
    *,
    labeled_batch_size: int,
    unlabeled_batch_size: int,
) -> None:
    """algorithm이 필요로 할 때만 labeled/unlabeled batch 크기를 전달한다."""

    if labeled_batch_size < 0:
        raise ValueError("labeled_batch_size must not be negative.")
    if unlabeled_batch_size <= 0:
        raise ValueError("unlabeled_batch_size must be positive.")
    configure_batching = getattr(algorithm, "configure_batching", None)
    if callable(configure_batching):
        configure_batching(
            labeled_batch_size=labeled_batch_size,
            unlabeled_batch_size=unlabeled_batch_size,
        )


def configure_query_ssl_algorithm_dataset(
    algorithm: QuerySslAlgorithm,
    *,
    num_classes: int,
    unlabeled_row_count: int,
) -> None:
    """algorithm이 필요로 할 때만 dataset-level state 크기를 전달한다."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if unlabeled_row_count <= 0:
        raise ValueError("unlabeled_row_count must be positive.")
    configure_dataset = getattr(algorithm, "configure_dataset", None)
    if callable(configure_dataset):
        configure_dataset(
            num_classes=num_classes,
            unlabeled_row_count=unlabeled_row_count,
        )


def configure_query_ssl_algorithm_model(
    algorithm: QuerySslAlgorithm,
    *,
    model: TextBatchClassifier,
    device: TorchDevice,
) -> None:
    """algorithm이 필요로 할 때만 model-dependent runtime state를 준비한다."""

    configure_model = getattr(algorithm, "configure_model", None)
    if callable(configure_model):
        configure_model(model=model, device=device)


def compute_query_ssl_algorithm_step(
    algorithm: QuerySslAlgorithm,
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Any],
    step_context: QuerySslStepContext | None = None,
) -> QuerySslStepResult:
    """context-aware algorithm은 context seam으로, 기존 algorithm은 그대로 호출한다."""

    compute_step_with_context = getattr(algorithm, "compute_step_with_context", None)
    if step_context is not None and callable(compute_step_with_context):
        return compute_step_with_context(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            step_context=step_context,
        )
    return algorithm.compute_step(
        model=model,
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
    )


def after_query_ssl_optimizer_step(
    algorithm: QuerySslAlgorithm,
    *,
    model: TextBatchClassifier,
    step_context: QuerySslStepContext,
) -> None:
    """optimizer step 이후 필요한 algorithm runtime hook을 실행한다."""

    after_optimizer_step = getattr(algorithm, "after_optimizer_step", None)
    if callable(after_optimizer_step):
        after_optimizer_step(model=model, step_context=step_context)

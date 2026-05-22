"""LoRA-classifier logical partition delta helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import Tensor, nn

from methods.adaptation.lora_classifier.update.partitioned_delta import (
    LoraClassifierPartitionDelta,
)


def named_trainable_parameter_tensors(model: nn.Module) -> dict[str, Tensor]:
    """현재 autograd graph에 연결된 trainable parameter mapping을 반환한다."""

    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def snapshot_trainable_parameter_tensors(model: nn.Module) -> dict[str, Tensor]:
    """trainable parameter 값을 detached clone으로 저장한다."""

    return {
        name: parameter.detach().clone()
        for name, parameter in named_trainable_parameter_tensors(model).items()
    }


def diff_parameter_snapshots(
    *,
    after: Mapping[str, Tensor],
    before: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """동일한 parameter key/shape의 snapshot 차이를 계산한다."""

    _validate_matching_snapshot_keys(after=after, before=before)
    deltas: dict[str, Tensor] = {}
    for name in sorted(after):
        if after[name].shape != before[name].shape:
            raise ValueError(f"Parameter snapshot shape mismatch for {name!r}.")
        deltas[name] = after[name] - before[name].to(
            device=after[name].device,
            dtype=after[name].dtype,
        )
    return deltas


def build_lora_classifier_partition_delta_from_parameter_deltas(
    *,
    partition_name: str,
    parameter_deltas: Mapping[str, Tensor],
    labels: Sequence[str],
) -> LoraClassifierPartitionDelta:
    """trainable parameter delta snapshot을 LoRA/head payload partition으로 바꾼다."""

    normalized_labels = _normalize_labels(labels)
    lora_parameter_deltas: dict[str, list[float]] = {}
    classifier_head_weight_deltas: dict[str, list[float]] = {}
    classifier_head_bias_deltas: dict[str, float] = {}

    for name, delta in sorted(parameter_deltas.items()):
        detached = delta.detach().cpu()
        if name == "classifier.weight":
            _validate_classifier_weight_delta(
                delta=detached,
                labels=normalized_labels,
            )
            for label_index, label in enumerate(normalized_labels):
                classifier_head_weight_deltas[label] = [
                    float(value) for value in detached[label_index].reshape(-1).tolist()
                ]
            continue
        if name == "classifier.bias":
            _validate_classifier_bias_delta(delta=detached, labels=normalized_labels)
            classifier_head_bias_deltas = {
                label: float(detached[label_index].item())
                for label_index, label in enumerate(normalized_labels)
            }
            continue
        lora_parameter_deltas[name] = [
            float(value) for value in detached.reshape(-1).tolist()
        ]

    return LoraClassifierPartitionDelta(
        partition_name=partition_name,
        lora_parameter_deltas=lora_parameter_deltas,
        classifier_head_weight_deltas=classifier_head_weight_deltas,
        classifier_head_bias_deltas=classifier_head_bias_deltas,
    )


def _validate_matching_snapshot_keys(
    *,
    after: Mapping[str, Tensor],
    before: Mapping[str, Tensor],
) -> None:
    after_keys = set(after)
    before_keys = set(before)
    if after_keys == before_keys:
        return
    raise ValueError(
        "Parameter snapshots must contain the same trainable keys: "
        f"missing_after={sorted(before_keys - after_keys)}, "
        f"missing_before={sorted(after_keys - before_keys)}"
    )


def _normalize_labels(labels: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(label).strip() for label in labels if str(label).strip())
    if not normalized:
        raise ValueError("labels must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("labels must not contain duplicates.")
    return normalized


def _validate_classifier_weight_delta(
    *,
    delta: Tensor,
    labels: Sequence[str],
) -> None:
    if delta.ndim != 2:
        raise ValueError("classifier.weight delta must be a 2D tensor.")
    if delta.shape[0] != len(labels):
        raise ValueError(
            "classifier.weight first dimension must match label schema size."
        )


def _validate_classifier_bias_delta(
    *,
    delta: Tensor,
    labels: Sequence[str],
) -> None:
    if delta.ndim != 1:
        raise ValueError("classifier.bias delta must be a 1D tensor.")
    if delta.shape[0] != len(labels):
        raise ValueError("classifier.bias dimension must match label schema size.")

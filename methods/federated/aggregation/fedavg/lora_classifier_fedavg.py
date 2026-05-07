"""LoRA-classifier family용 FedAvg 계산 core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from methods.federated.aggregation.fedavg.fedavg import (
    WeightedScalarMappingUpdate,
    WeightedScalarUpdate,
    WeightedVectorMappingUpdate,
    weighted_average_scalar_mappings,
    weighted_average_scalars,
    weighted_average_vector_mappings,
)
from methods.federated.aggregation.registry import register_federated_aggregation_method
from shared.src.contracts.adapter_family_metadata import LORA_CLASSIFIER_FAMILY_METADATA


@dataclass(frozen=True, slots=True)
class LoraClassifierFedAvgUpdate:
    """main_server boundary와 분리된 LoRA-classifier FedAvg 입력."""

    lora_parameter_deltas: Mapping[str, Sequence[float]]
    classifier_head_weight_deltas: Mapping[str, Sequence[float]]
    classifier_head_bias_deltas: Mapping[str, float]
    example_count: int
    mean_confidence: float | None
    mean_margin: float | None
    delta_l2_norm: float


@dataclass(frozen=True, slots=True)
class LoraClassifierFedAvgResult:
    """LoRA-classifier FedAvg 계산 결과."""

    lora_parameter_deltas: dict[str, list[float]]
    classifier_head_weight_deltas: dict[str, list[float]]
    classifier_head_bias_deltas: dict[str, float]
    aggregated_metrics: dict[str, float]
    update_count: int


@register_federated_aggregation_method(
    adapter_kind=LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind,
    method_name="fedavg",
    aliases=("lora_classifier_fedavg",),
)
def compute_lora_classifier_fedavg(
    *,
    label_schema: Sequence[str],
    updates: Sequence[LoraClassifierFedAvgUpdate],
) -> LoraClassifierFedAvgResult:
    """LoRA parameter delta와 classifier-head delta를 example_count로 평균한다."""

    labels = _normalize_label_schema(label_schema)
    valid_updates = tuple(update for update in updates if update.example_count > 0)
    if not valid_updates:
        raise ValueError("At least one non-empty LoRA-classifier update is required.")

    lora_parameter_deltas = weighted_average_vector_mappings(
        [
            WeightedVectorMappingUpdate(
                values=update.lora_parameter_deltas,
                weight=float(update.example_count),
            )
            for update in valid_updates
        ]
    )
    classifier_head_weight_deltas = weighted_average_vector_mappings(
        [
            WeightedVectorMappingUpdate(
                values=_normalize_classifier_head_weight_deltas(
                    update,
                    labels=labels,
                ),
                weight=float(update.example_count),
            )
            for update in valid_updates
        ]
    )
    classifier_head_bias_deltas = weighted_average_scalar_mappings(
        [
            WeightedScalarMappingUpdate(
                values=_normalize_classifier_head_bias_deltas(
                    update,
                    labels=labels,
                ),
                weight=float(update.example_count),
            )
            for update in valid_updates
        ]
    )

    return LoraClassifierFedAvgResult(
        lora_parameter_deltas=lora_parameter_deltas,
        classifier_head_weight_deltas=classifier_head_weight_deltas,
        classifier_head_bias_deltas=classifier_head_bias_deltas,
        aggregated_metrics={
            **_aggregate_common_metrics(valid_updates),
            "lora_parameter_count": float(len(lora_parameter_deltas)),
            "classifier_head_label_count": float(len(classifier_head_weight_deltas)),
        },
        update_count=len(valid_updates),
    )


def _normalize_label_schema(label_schema: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(str(label).strip() for label in label_schema if str(label).strip())
    if not labels:
        raise ValueError("label_schema must not be empty.")
    if len(set(labels)) != len(labels):
        raise ValueError("label_schema must not contain duplicates.")
    return labels


def _normalize_classifier_head_weight_deltas(
    update: LoraClassifierFedAvgUpdate,
    *,
    labels: Sequence[str],
) -> dict[str, Sequence[float]]:
    if set(update.classifier_head_weight_deltas) != set(labels):
        raise ValueError(
            "LoRA-classifier FedAvg classifier head weight delta keys must match "
            "label_schema."
        )
    return {
        label: update.classifier_head_weight_deltas[label]
        for label in sorted(update.classifier_head_weight_deltas)
    }


def _normalize_classifier_head_bias_deltas(
    update: LoraClassifierFedAvgUpdate,
    *,
    labels: Sequence[str],
) -> dict[str, float]:
    extra_labels = set(update.classifier_head_bias_deltas) - set(labels)
    if extra_labels:
        raise ValueError(
            "LoRA-classifier FedAvg bias deltas contain unknown labels: "
            f"{sorted(extra_labels)}"
        )
    return {
        label: float(update.classifier_head_bias_deltas.get(label, 0.0))
        for label in labels
    }


def _aggregate_common_metrics(
    updates: Sequence[LoraClassifierFedAvgUpdate],
) -> dict[str, float]:
    return {
        "client_count": float(len(updates)),
        "example_count": float(sum(update.example_count for update in updates)),
        "mean_confidence": weighted_average_scalars(
            [
                WeightedScalarUpdate(
                    value=update.mean_confidence or 0.0,
                    weight=float(update.example_count),
                )
                for update in updates
            ]
        ),
        "mean_margin": weighted_average_scalars(
            [
                WeightedScalarUpdate(
                    value=update.mean_margin or 0.0,
                    weight=float(update.example_count),
                )
                for update in updates
            ]
        ),
        "mean_delta_l2_norm": weighted_average_scalars(
            [
                WeightedScalarUpdate(
                    value=update.delta_l2_norm,
                    weight=float(update.example_count),
                )
                for update in updates
            ]
        ),
    }

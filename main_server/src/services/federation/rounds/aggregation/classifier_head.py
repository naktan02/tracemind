"""Classifier-head aggregation backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from main_server.src.services.federation.rounds.aggregation.models import (
    AggregationResult,
)
from shared.src.config.adapter_family_metadata import CLASSIFIER_HEAD_FAMILY_METADATA
from shared.src.contracts.adapter_contracts import (
    ClassifierHeadDelta,
    ClassifierHeadState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)


@dataclass(slots=True)
class ClassifierHeadFedAvgAggregationService:
    """Classifier-head delta를 전역 선형 head 상태로 집계한다."""

    adapter_kind: str = CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ) -> AggregationResult:
        if not isinstance(base_state, ClassifierHeadState):
            raise TypeError(
                "ClassifierHeadFedAvgAggregationService expects "
                f"ClassifierHeadState as the base state, got {type(base_state)!r}."
            )
        if base_state.adapter_kind != self.adapter_kind:
            raise ValueError(
                "Base state adapter_kind does not match the classifier-head "
                f"aggregator: {base_state.adapter_kind}"
            )

        valid_updates = [
            payload for payload in update_payloads if payload.example_count > 0
        ]
        if not valid_updates:
            raise ValueError("At least one non-empty update payload is required.")

        labels = base_state.labels
        embedding_dim = base_state.embedding_dim
        total_examples = sum(payload.example_count for payload in valid_updates)
        weighted_weight_deltas = {
            label: [0.0] * embedding_dim for label in labels
        }
        weighted_bias_deltas = {label: 0.0 for label in labels}
        weighted_confidence = 0.0
        weighted_margin = 0.0
        weighted_delta_norm = 0.0

        for payload in valid_updates:
            if not isinstance(payload, ClassifierHeadDelta):
                raise TypeError(
                    "ClassifierHeadFedAvgAggregationService expects "
                    f"ClassifierHeadDelta updates, got {type(payload)!r}."
                )
            if payload.adapter_kind != self.adapter_kind:
                raise ValueError(
                    "Update adapter_kind does not match the classifier-head "
                    f"aggregator: {payload.adapter_kind}"
                )
            if payload.model_id != base_state.model_id:
                raise ValueError("All update payloads must match the base model_id.")
            if payload.base_model_revision != base_state.model_revision:
                raise ValueError(
                    "All update payloads must match the base model revision."
                )
            if payload.training_scope != base_state.training_scope:
                raise ValueError("All update payloads must match the training scope.")
            if payload.labels != labels:
                raise ValueError(
                    "Classifier head updates must share the same ordered labels."
                )
            if payload.embedding_dim != embedding_dim:
                raise ValueError(
                    "All update payloads must share the same embedding_dim."
                )

            weight = payload.example_count / total_examples
            weighted_confidence += payload.mean_confidence * weight
            weighted_margin += (payload.mean_margin or 0.0) * weight
            weighted_delta_norm += payload.l2_norm() * weight
            for label in labels:
                for index, value in enumerate(payload.label_weight_deltas[label]):
                    weighted_weight_deltas[label][index] += float(value) * weight
                weighted_bias_deltas[label] += (
                    float(payload.label_bias_deltas.get(label, 0.0)) * weight
                )

        next_state = ClassifierHeadState(
            schema_version=base_state.schema_version,
            adapter_kind=base_state.adapter_kind,
            model_id=base_state.model_id,
            model_revision=next_model_revision,
            training_scope=base_state.training_scope,
            updated_at=aggregated_at,
            label_weights={
                label: [
                    float(base_value) + float(delta)
                    for base_value, delta in zip(
                        base_state.label_weights[label],
                        weighted_weight_deltas[label],
                        strict=True,
                    )
                ]
                for label in labels
            },
            label_biases={
                label: float(base_state.label_biases.get(label, 0.0))
                + float(weighted_bias_deltas[label])
                for label in labels
            },
        )
        return AggregationResult(
            next_state=next_state,
            aggregated_metrics={
                "client_count": float(len(valid_updates)),
                "example_count": float(total_examples),
                "mean_confidence": weighted_confidence,
                "mean_margin": weighted_margin,
                "mean_delta_l2_norm": weighted_delta_norm,
            },
            update_count=len(valid_updates),
        )

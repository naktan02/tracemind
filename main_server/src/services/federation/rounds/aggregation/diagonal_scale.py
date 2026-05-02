"""Diagonal-scale aggregation backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from shared.src.config.adapter_family_metadata import DIAGONAL_SCALE_FAMILY_METADATA
from shared.src.contracts.adapter_contracts import (
    VectorAdapterDelta,
    VectorAdapterState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from .diagonal_scale_defaults import DEFAULT_DIAGONAL_SCALE_FEDAVG_AGGREGATION_CONFIG
from .models import AggregationConfig, AggregationResult


@dataclass(slots=True)
class DiagonalScaleAggregationService:
    """Diagonal scale adapter update를 전역 상태로 집계한다."""

    adapter_kind: str = DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind
    config: AggregationConfig = field(
        default_factory=lambda: DEFAULT_DIAGONAL_SCALE_FEDAVG_AGGREGATION_CONFIG
    )

    @classmethod
    def from_mapping(
        cls,
        source,
    ) -> "DiagonalScaleAggregationService":
        return cls(config=AggregationConfig.from_mapping(source))

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ) -> AggregationResult:
        if not isinstance(base_state, VectorAdapterState):
            raise TypeError(
                "DiagonalScaleAggregationService expects VectorAdapterState as the "
                f"base state, got {type(base_state)!r}."
            )
        if base_state.adapter_kind != self.adapter_kind:
            raise ValueError(
                "Base state adapter_kind does not match the diagonal scale "
                f"aggregator: {base_state.adapter_kind}"
            )

        valid_updates = [
            payload for payload in update_payloads if payload.example_count > 0
        ]
        if not valid_updates:
            raise ValueError("At least one non-empty update payload is required.")

        embedding_dim = base_state.embedding_dim
        total_examples = sum(payload.example_count for payload in valid_updates)
        weighted_delta = [0.0] * embedding_dim
        weighted_confidence = 0.0
        weighted_margin = 0.0
        weighted_delta_norm = 0.0

        for payload in valid_updates:
            if not isinstance(payload, VectorAdapterDelta):
                raise TypeError(
                    "DiagonalScaleAggregationService expects VectorAdapterDelta "
                    f"updates, got {type(payload)!r}."
                )
            if payload.adapter_kind != self.adapter_kind:
                raise ValueError(
                    "Update adapter_kind does not match the diagonal scale "
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
            if payload.embedding_dim != embedding_dim:
                raise ValueError(
                    "All update payloads must share the same embedding_dim."
                )

            weight = payload.example_count / total_examples
            weighted_confidence += payload.mean_confidence * weight
            weighted_margin += (payload.mean_margin or 0.0) * weight
            weighted_delta_norm += payload.l2_norm() * weight
            for index, value in enumerate(payload.dimension_deltas):
                weighted_delta[index] += value * weight

        next_scales = [
            max(
                self.config.min_scale,
                min(self.config.max_scale, scale + delta),
            )
            for scale, delta in zip(
                base_state.dimension_scales,
                weighted_delta,
                strict=True,
            )
        ]
        next_state = VectorAdapterState(
            schema_version=base_state.schema_version,
            model_id=base_state.model_id,
            model_revision=next_model_revision,
            training_scope=base_state.training_scope,
            dimension_scales=next_scales,
            updated_at=aggregated_at,
            adapter_kind=base_state.adapter_kind,
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

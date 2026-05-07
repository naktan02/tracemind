"""Diagonal-scale aggregation backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from methods.federated.aggregation.fedavg.diagonal_scale_fedavg import (
    DiagonalScaleFedAvgUpdate,
    compute_diagonal_scale_fedavg,
)
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
from .runtime_adapter import (
    require_base_adapter_kind,
    require_update_matches_base,
    select_non_empty_updates,
)


@dataclass(slots=True)
class DiagonalScaleAggregationService:
    """Diagonal scale server boundary를 FedAvg method core에 연결한다."""

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
        require_base_adapter_kind(
            base_state=base_state,
            adapter_kind=self.adapter_kind,
            context="diagonal scale",
        )

        valid_updates = select_non_empty_updates(update_payloads)

        embedding_dim = base_state.embedding_dim
        method_updates: list[DiagonalScaleFedAvgUpdate] = []

        for payload in valid_updates:
            if not isinstance(payload, VectorAdapterDelta):
                raise TypeError(
                    "DiagonalScaleAggregationService expects VectorAdapterDelta "
                    f"updates, got {type(payload)!r}."
                )
            require_update_matches_base(
                payload=payload,
                base_state=base_state,
                adapter_kind=self.adapter_kind,
                context="diagonal scale",
            )
            if payload.embedding_dim != embedding_dim:
                raise ValueError(
                    "All update payloads must share the same embedding_dim."
                )
            method_updates.append(
                DiagonalScaleFedAvgUpdate(
                    dimension_deltas=payload.dimension_deltas,
                    example_count=payload.example_count,
                    mean_confidence=payload.mean_confidence,
                    mean_margin=payload.mean_margin,
                    delta_l2_norm=payload.l2_norm(),
                )
            )

        method_result = compute_diagonal_scale_fedavg(
            base_dimension_scales=base_state.dimension_scales,
            updates=method_updates,
            min_scale=self.config.min_scale,
            max_scale=self.config.max_scale,
        )
        next_state = VectorAdapterState(
            schema_version=base_state.schema_version,
            model_id=base_state.model_id,
            model_revision=next_model_revision,
            training_scope=base_state.training_scope,
            dimension_scales=method_result.next_dimension_scales,
            updated_at=aggregated_at,
            adapter_kind=base_state.adapter_kind,
        )
        return AggregationResult(
            next_state=next_state,
            aggregated_metrics=method_result.aggregated_metrics,
            update_count=method_result.update_count,
        )

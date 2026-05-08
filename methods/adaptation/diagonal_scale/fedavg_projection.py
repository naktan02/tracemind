"""Diagonal-scale payload projection for FedAvg aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from methods.federated.aggregation.base import (
    AggregationConfigScalar,
    FederatedAggregationContext,
    FederatedAggregationResult,
)
from methods.federated.aggregation.fedavg.diagonal_scale_fedavg import (
    DiagonalScaleFedAvgUpdate,
    compute_diagonal_scale_fedavg,
)
from methods.federated.aggregation.fedavg.strategy import (
    FedAvgAdapterStrategySpec,
    register_fedavg_adapter_strategy,
)
from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    DIAGONAL_SCALE_ADAPTER_KIND,
)
from shared.src.contracts.adapter_contracts import (
    VectorAdapterDelta,
    VectorAdapterState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

DEFAULT_DIAGONAL_SCALE_MIN_SCALE = 0.75
DEFAULT_DIAGONAL_SCALE_MAX_SCALE = 1.25


def aggregate_diagonal_scale_fedavg(
    base_state: SharedAdapterState,
    update_payloads: Sequence[SharedAdapterUpdate],
    context: FederatedAggregationContext,
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> FederatedAggregationResult:
    """Diagonal-scale update payload를 FedAvg core 입력으로 변환한다."""

    base_state = cast(VectorAdapterState, base_state)
    updates = [cast(VectorAdapterDelta, payload) for payload in update_payloads]
    embedding_dim = base_state.embedding_dim
    method_updates: list[DiagonalScaleFedAvgUpdate] = []

    for payload in updates:
        if payload.embedding_dim != embedding_dim:
            raise ValueError("All update payloads must share the same embedding_dim.")
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
        min_scale=_read_float(
            overrides,
            "min_scale",
            DEFAULT_DIAGONAL_SCALE_MIN_SCALE,
        ),
        max_scale=_read_float(
            overrides,
            "max_scale",
            DEFAULT_DIAGONAL_SCALE_MAX_SCALE,
        ),
    )
    next_state = VectorAdapterState(
        schema_version=base_state.schema_version,
        model_id=base_state.model_id,
        model_revision=context.next_model_revision,
        training_scope=base_state.training_scope,
        dimension_scales=method_result.next_dimension_scales,
        updated_at=context.aggregated_at,
        adapter_kind=base_state.adapter_kind,
    )
    return FederatedAggregationResult(
        next_state=next_state,
        aggregated_metrics=method_result.aggregated_metrics,
        update_count=method_result.update_count,
    )


def _read_float(
    source: Mapping[str, AggregationConfigScalar] | None,
    key: str,
    default: float,
) -> float:
    if source is None:
        return default
    value = source.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must not be bool.")
    return float(value)


register_fedavg_adapter_strategy(
    FedAvgAdapterStrategySpec(
        adapter_kind=DIAGONAL_SCALE_ADAPTER_KIND,
        state_type=VectorAdapterState,
        update_type=VectorAdapterDelta,
        context="diagonal scale",
        aliases=("diagonal_scale_fedavg",),
        implementation_module=compute_diagonal_scale_fedavg.__module__,
        core_function_name=compute_diagonal_scale_fedavg.__name__,
        metadata={"adapter_kind": DIAGONAL_SCALE_ADAPTER_KIND},
        aggregate=aggregate_diagonal_scale_fedavg,
    )
)

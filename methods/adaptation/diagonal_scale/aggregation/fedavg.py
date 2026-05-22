"""Diagonal-scale family용 FedAvg 계산 core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from methods.common.config_reading import read_float
from methods.federated.aggregation.base import (
    AggregationConfigScalar,
    FederatedAggregationContext,
    FederatedAggregationResult,
)
from methods.federated.aggregation.fedavg.strategy import (
    FedAvgAdapterStrategySpec,
    register_fedavg_adapter_strategy,
)
from methods.federated.aggregation.fedavg.update_metrics import (
    FedAvgObservationMetricUpdate,
    aggregate_update_observation_metrics,
)
from methods.federated.aggregation.fedavg.weighted_average import (
    WeightedVectorUpdate,
    weighted_average_vectors,
)
from methods.federated.aggregation_weighting import (
    AGGREGATION_WEIGHT_EXAMPLE_COUNT,
    AggregationWeightPolicy,
    aggregation_weight_for_update,
)
from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    DIAGONAL_SCALE_ADAPTER_KIND,
    VectorAdapterDelta,
    VectorAdapterState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)


@dataclass(frozen=True, slots=True)
class DiagonalScaleFedAvgUpdate:
    """main_server boundary와 분리된 diagonal-scale FedAvg 입력."""

    dimension_deltas: Sequence[float]
    example_count: int
    mean_confidence: float
    mean_margin: float | None
    delta_l2_norm: float


@dataclass(frozen=True, slots=True)
class DiagonalScaleFedAvgResult:
    """diagonal-scale FedAvg 계산 결과."""

    next_dimension_scales: list[float]
    aggregated_metrics: dict[str, float]
    update_count: int


def compute_diagonal_scale_fedavg(
    *,
    base_dimension_scales: Sequence[float],
    updates: Sequence[DiagonalScaleFedAvgUpdate],
    min_scale: float,
    max_scale: float,
    weight_policy_name: str = AGGREGATION_WEIGHT_EXAMPLE_COUNT,
) -> DiagonalScaleFedAvgResult:
    """차원별 scale delta를 policy weight로 평균하고 clamp한다."""

    if min_scale > max_scale:
        raise ValueError("min_scale must be less than or equal to max_scale.")

    base_scales = [float(scale) for scale in base_dimension_scales]
    if not base_scales:
        raise ValueError("base_dimension_scales must not be empty.")

    valid_updates = tuple(update for update in updates if update.example_count > 0)
    weight_policy = AggregationWeightPolicy(name=weight_policy_name)
    weighted_delta = weighted_average_vectors(
        [
            WeightedVectorUpdate(
                values=update.dimension_deltas,
                weight=aggregation_weight_for_update(update, policy=weight_policy),
            )
            for update in valid_updates
        ]
    )
    if len(weighted_delta) != len(base_scales):
        raise ValueError("FedAvg delta dimension must match base_dimension_scales.")

    next_dimension_scales = [
        max(min_scale, min(max_scale, scale + delta))
        for scale, delta in zip(base_scales, weighted_delta, strict=True)
    ]
    return DiagonalScaleFedAvgResult(
        next_dimension_scales=next_dimension_scales,
        aggregated_metrics=_aggregate_common_metrics(valid_updates),
        update_count=len(valid_updates),
    )


def _aggregate_common_metrics(
    updates: Sequence[DiagonalScaleFedAvgUpdate],
) -> dict[str, float]:
    return aggregate_update_observation_metrics(
        [
            FedAvgObservationMetricUpdate(
                example_count=update.example_count,
                mean_confidence=update.mean_confidence,
                mean_margin=update.mean_margin,
                delta_l2_norm=update.delta_l2_norm,
            )
            for update in updates
        ]
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
        min_scale=read_float(
            overrides,
            "min_scale",
            DEFAULT_DIAGONAL_SCALE_MIN_SCALE,
        ),
        max_scale=read_float(
            overrides,
            "max_scale",
            DEFAULT_DIAGONAL_SCALE_MAX_SCALE,
        ),
        weight_policy_name=str((overrides or {}).get("weight_policy", "example_count")),
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

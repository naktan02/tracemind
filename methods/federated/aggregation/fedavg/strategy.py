"""Generic FedAvg strategy wiring."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from methods.federated.aggregation.base import (
    AggregationConfigScalar,
    FederatedAggregationContext,
    FederatedAggregationResult,
    FederatedAggregationStrategy,
)
from methods.federated.aggregation.registry import (
    register_federated_aggregation_strategy,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

StateT = TypeVar("StateT", bound=SharedAdapterState)
UpdateT = TypeVar("UpdateT", bound=SharedAdapterUpdate)

FedAvgAggregate = Callable[
    [
        SharedAdapterState,
        Sequence[SharedAdapterUpdate],
        FederatedAggregationContext,
        Mapping[str, AggregationConfigScalar] | None,
    ],
    FederatedAggregationResult,
]


@dataclass(frozen=True, slots=True)
class FedAvgAdapterStrategySpec:
    """FedAvg가 한 adapter family payload를 처리하는 method-owned spec."""

    adapter_kind: str
    state_type: type[object]
    update_type: type[object]
    context: str
    aliases: tuple[str, ...]
    implementation_module: str
    core_function_name: str
    metadata: Mapping[str, AggregationConfigScalar | None]
    aggregate: FedAvgAggregate
    method_name: str = "fedavg"


@dataclass(slots=True)
class FedAvgAggregationStrategy:
    """Adapter family contract를 FedAvg method core에 연결하는 strategy."""

    spec: FedAvgAdapterStrategySpec
    overrides: Mapping[str, AggregationConfigScalar] | None = None

    @property
    def method_name(self) -> str:
        """registry에서 선택한 aggregation backend 이름."""

        return self.spec.method_name

    @property
    def adapter_kind(self) -> str:
        """이 strategy가 처리하는 shared adapter family discriminator."""

        return self.spec.adapter_kind

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        context: FederatedAggregationContext,
    ) -> FederatedAggregationResult:
        """공통 lineage/type 검증 후 family별 FedAvg projection을 호출한다."""

        typed_base_state = _require_typed_base_state(
            base_state=base_state,
            state_type=self.spec.state_type,
            adapter_kind=self.spec.adapter_kind,
            context=self.spec.context,
            strategy_name="FedAvgAggregationStrategy",
        )
        typed_updates = _select_validated_updates(
            update_payloads,
            update_type=self.spec.update_type,
            base_state=typed_base_state,
            adapter_kind=self.spec.adapter_kind,
            context=self.spec.context,
            strategy_name="FedAvgAggregationStrategy",
        )
        return self.spec.aggregate(
            typed_base_state,
            typed_updates,
            context,
            self.overrides,
        )


def register_fedavg_adapter_strategy(spec: FedAvgAdapterStrategySpec) -> None:
    """adapter family projection module에서 FedAvg strategy를 등록한다."""

    register_federated_aggregation_strategy(
        adapter_kind=spec.adapter_kind,
        method_name=spec.method_name,
        aliases=spec.aliases,
        implementation_module=spec.implementation_module,
        core_function_name=spec.core_function_name,
        metadata=spec.metadata,
        factory=_build_fedavg_strategy_factory(spec),
    )


def _build_fedavg_strategy_factory(
    spec: FedAvgAdapterStrategySpec,
) -> Callable[
    [Mapping[str, AggregationConfigScalar] | None],
    FederatedAggregationStrategy,
]:
    def _factory(
        overrides: Mapping[str, AggregationConfigScalar] | None,
    ) -> FedAvgAggregationStrategy:
        return FedAvgAggregationStrategy(spec=spec, overrides=overrides)

    return _factory


def _require_typed_base_state(
    *,
    base_state: SharedAdapterState,
    state_type: type[StateT],
    adapter_kind: str,
    context: str,
    strategy_name: str,
) -> StateT:
    if not isinstance(base_state, state_type):
        raise TypeError(
            f"{strategy_name} expects {state_type.__name__} as the base state, "
            f"got {type(base_state)!r}."
        )
    if base_state.adapter_kind != adapter_kind:
        raise ValueError(
            f"Base state adapter_kind does not match the {context} aggregator: "
            f"{base_state.adapter_kind}"
        )
    return base_state


def _select_validated_updates(
    update_payloads: Sequence[SharedAdapterUpdate],
    *,
    update_type: type[UpdateT],
    base_state: SharedAdapterState,
    adapter_kind: str,
    context: str,
    strategy_name: str,
) -> list[UpdateT]:
    valid_updates = [
        payload for payload in update_payloads if payload.example_count > 0
    ]
    if not valid_updates:
        raise ValueError("At least one non-empty update payload is required.")

    typed_updates: list[UpdateT] = []
    for payload in valid_updates:
        if not isinstance(payload, update_type):
            raise TypeError(
                f"{strategy_name} expects {update_type.__name__} updates, "
                f"got {type(payload)!r}."
            )
        _require_update_matches_base(
            payload=payload,
            base_state=base_state,
            adapter_kind=adapter_kind,
            context=context,
        )
        typed_updates.append(payload)
    return typed_updates


def _require_update_matches_base(
    *,
    payload: SharedAdapterUpdate,
    base_state: SharedAdapterState,
    adapter_kind: str,
    context: str,
) -> None:
    if payload.adapter_kind != adapter_kind:
        raise ValueError(
            f"Update adapter_kind does not match the {context} aggregator: "
            f"{payload.adapter_kind}"
        )
    if payload.model_id != base_state.model_id:
        raise ValueError("All update payloads must match the base model_id.")
    if payload.base_model_revision != base_state.model_revision:
        raise ValueError("All update payloads must match the base model revision.")
    if payload.training_scope != base_state.training_scope:
        raise ValueError("All update payloads must match the training scope.")

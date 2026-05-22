"""FL SSL method descriptor를 simulation runtime으로 연결하는 adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from methods.federated_ssl.base import (
    TRAINING_ROW_SOURCE_UNLABELED_POOL_WHEN_AVAILABLE,
    TRAINING_ROW_SOURCES,
    FederatedSslMethodDescriptor,
)
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientShard,
)
from scripts.runtime_adapters.federated_server.round_request_mapper import (
    build_round_open_request,
)
from scripts.runtime_adapters.federated_server.task_config_surface import (
    FederatedTrainingTaskConfig,
)
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

MANUAL_BASELINE_RUNTIME_NAME = "manual_baseline"
MANUAL_BASELINE_TRAINING_TASK_TYPE = TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING.value
MANUAL_BASELINE_TRAINING_ROW_SOURCE = TRAINING_ROW_SOURCE_UNLABELED_POOL_WHEN_AVAILABLE


@dataclass(frozen=True, slots=True)
class FederatedClientLocalTrainingContext:
    """client local training 준비에 필요한 runtime 입력."""

    shard: FederatedClientShard
    training_task: Any


@dataclass(frozen=True, slots=True)
class FederatedClientLocalTrainingPlan:
    """runtime이 선택한 local training 입력과 실행 adapter."""

    rows: list[LabeledQueryRow]
    examples: tuple[Any, ...]
    service: Any


class FederatedSslSimulationRuntime(Protocol):
    """Simulation loop가 호출하는 FL SSL 실행 조합."""

    descriptor: FederatedSslMethodDescriptor | None
    runtime_name: str
    training_task_type: str
    training_row_source: str

    def build_round_open_request(
        self,
        *,
        round_id: str,
        training_task_config: FederatedTrainingTaskConfig,
    ) -> Any:
        """method별 round task를 생성한다."""

    def build_local_training_plan(
        self,
        *,
        context: FederatedClientLocalTrainingContext,
    ) -> FederatedClientLocalTrainingPlan:
        """client local training 입력과 실행 adapter를 준비한다."""


@dataclass(frozen=True, slots=True)
class DefaultFederatedSslSimulationRuntime:
    """기본 FL SSL simulation runtime 조합."""

    runtime_name: str
    training_task_type: str
    training_row_source: str
    descriptor: FederatedSslMethodDescriptor | None = None

    def __post_init__(self) -> None:
        if self.training_row_source not in TRAINING_ROW_SOURCES:
            raise ValueError(
                f"training_row_source must be one of {sorted(TRAINING_ROW_SOURCES)}."
            )

    def build_round_open_request(
        self,
        *,
        round_id: str,
        training_task_config: FederatedTrainingTaskConfig,
    ) -> Any:
        return build_round_open_request(
            round_id=round_id,
            training_task_config=training_task_config,
        )

    def build_local_training_plan(
        self,
        *,
        context: FederatedClientLocalTrainingContext,
    ) -> FederatedClientLocalTrainingPlan:
        del context
        raise NotImplementedError(
            "FL SSL simulation no longer supports prototype-scored generic local "
            "training. Use the LoRA-classifier method/manual local objective path."
        )


def build_federated_ssl_simulation_runtime(
    name: str,
) -> FederatedSslSimulationRuntime:
    """method descriptor를 기본 simulation runtime adapter로 변환한다."""
    descriptor = resolve_federated_ssl_method_descriptor(name)
    if not descriptor.runtime_capabilities.simulation_supported:
        raise NotImplementedError(
            "Federated SSL method is not available in simulation runtime: "
            f"{descriptor.name}"
        )
    if descriptor.runtime_capabilities.requires_custom_server_runtime:
        raise NotImplementedError(
            "Federated SSL method requires a custom server simulation runtime: "
            f"{descriptor.name}"
        )
    return DefaultFederatedSslSimulationRuntime(
        runtime_name=descriptor.name,
        training_task_type=descriptor.local_step.step_name,
        training_row_source=descriptor.local_step.training_row_source,
        descriptor=descriptor,
    )


def build_manual_federated_ssl_simulation_runtime() -> FederatedSslSimulationRuntime:
    """manual baseline 조합용 기본 simulation runtime을 만든다."""

    return DefaultFederatedSslSimulationRuntime(
        runtime_name=MANUAL_BASELINE_RUNTIME_NAME,
        training_task_type=MANUAL_BASELINE_TRAINING_TASK_TYPE,
        training_row_source=MANUAL_BASELINE_TRAINING_ROW_SOURCE,
    )

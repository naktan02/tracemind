"""FL SSL method descriptor를 simulation runtime으로 연결하는 adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from methods.federated_ssl.base import (
    TRAINING_ROW_SOURCE_ALL_ROWS,
    TRAINING_ROW_SOURCE_LABELED_POOL_WHEN_AVAILABLE,
    TRAINING_ROW_SOURCE_UNLABELED_POOL_WHEN_AVAILABLE,
    TRAINING_ROW_SOURCES,
    FederatedSslMethodDescriptor,
)
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from scripts.experiments.fl_ssl.federated_simulation.adapters.evaluation import (
    build_training_examples,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientShard,
)
from scripts.runtime_adapters.federated_agent.training_runtime import (
    build_federated_local_training_service,
)
from scripts.runtime_adapters.federated_server.round_request_mapper import (
    build_round_open_request,
)
from scripts.runtime_adapters.federated_server.task_config_surface import (
    FederatedTrainingTaskConfig,
)
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter

MANUAL_BASELINE_RUNTIME_NAME = "manual_baseline"
MANUAL_BASELINE_TRAINING_TASK_TYPE = TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING.value
MANUAL_BASELINE_TRAINING_ROW_SOURCE = TRAINING_ROW_SOURCE_UNLABELED_POOL_WHEN_AVAILABLE


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

    def build_training_examples(
        self,
        *,
        rows: list[LabeledQueryRow],
        adapter: EmbeddingAdapter,
        adapter_state: SharedAdapterState,
        prototype_pack: PrototypePackPayload,
        model_id: str,
        scoring_service: Any,
        objective_config: TrainingObjectiveConfig,
    ) -> tuple[Any, ...]:
        """client shard row를 method별 local training 입력으로 변환한다."""

    def select_training_rows(
        self,
        *,
        shard: FederatedClientShard,
    ) -> list[LabeledQueryRow]:
        """client shard에서 이번 local step이 사용할 row를 고른다."""

    def build_local_training_service(
        self,
        *,
        client_state_root: Path,
        training_task: Any | None = None,
    ) -> Any:
        """client local trainer를 생성한다."""


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

    def build_training_examples(
        self,
        *,
        rows: list[LabeledQueryRow],
        adapter: EmbeddingAdapter,
        adapter_state: SharedAdapterState,
        prototype_pack: PrototypePackPayload,
        model_id: str,
        scoring_service: Any,
        objective_config: TrainingObjectiveConfig,
    ) -> tuple[Any, ...]:
        return build_training_examples(
            rows=rows,
            adapter=adapter,
            adapter_state=adapter_state,
            prototype_pack=prototype_pack,
            model_id=model_id,
            scoring_service=scoring_service,
            objective_config=objective_config,
        )

    def select_training_rows(
        self,
        *,
        shard: FederatedClientShard,
    ) -> list[LabeledQueryRow]:
        row_source = self.training_row_source
        if row_source == TRAINING_ROW_SOURCE_ALL_ROWS:
            return list(shard.rows)
        if (
            row_source == TRAINING_ROW_SOURCE_UNLABELED_POOL_WHEN_AVAILABLE
            and shard.client_pool_split_enforced
        ):
            return list(shard.unlabeled_rows)
        if (
            row_source == TRAINING_ROW_SOURCE_LABELED_POOL_WHEN_AVAILABLE
            and shard.client_pool_split_enforced
        ):
            return list(shard.labeled_rows)
        return list(shard.rows)

    def build_local_training_service(
        self,
        *,
        client_state_root: Path,
        training_task: Any | None = None,
    ) -> Any:
        return build_federated_local_training_service(
            client_state_root=client_state_root,
            training_task=training_task,
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
    if (
        descriptor.runtime_capabilities.requires_custom_client_runtime
        or descriptor.runtime_capabilities.requires_custom_server_runtime
    ):
        raise NotImplementedError(
            "Federated SSL method requires a custom simulation runtime: "
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

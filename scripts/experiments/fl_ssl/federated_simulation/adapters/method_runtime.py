"""FL SSL method descriptor를 simulation runtime으로 연결하는 adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from scripts.experiments.fl_ssl.federated_simulation.adapters.evaluation import (
    build_training_examples,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.task_config import (
    build_round_open_request,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedTrainingTaskConfig,
)
from scripts.io.labeled_query_rows import LabeledQueryRow
from scripts.runtime_adapters.federated_agent_runtime import (
    build_federated_local_training_service,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter


class FederatedSslSimulationRuntime(Protocol):
    """Simulation loop가 호출하는 FL SSL 실행 조합."""

    descriptor: FederatedSslMethodDescriptor

    def build_round_open_request(
        self,
        *,
        active_manifest: ModelManifest,
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

    def build_local_training_service(
        self,
        *,
        client_state_root: Path,
    ) -> Any:
        """client local trainer를 생성한다."""


@dataclass(frozen=True, slots=True)
class DefaultFederatedSslSimulationRuntime:
    """기본 FL SSL simulation runtime 조합."""

    descriptor: FederatedSslMethodDescriptor

    def build_round_open_request(
        self,
        *,
        active_manifest: ModelManifest,
        round_id: str,
        training_task_config: FederatedTrainingTaskConfig,
    ) -> Any:
        return build_round_open_request(
            active_manifest=active_manifest,
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

    def build_local_training_service(
        self,
        *,
        client_state_root: Path,
    ) -> Any:
        return build_federated_local_training_service(
            client_state_root=client_state_root
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
    return DefaultFederatedSslSimulationRuntime(descriptor=descriptor)

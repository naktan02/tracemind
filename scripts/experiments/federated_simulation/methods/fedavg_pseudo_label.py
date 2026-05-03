"""FedAvg pseudo-label FL SSL baseline runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.training.examples.models import EmbeddedTrainingExample
from agent.src.services.training.execution.local_training_service import (
    LocalTrainingService,
)
from main_server.src.services.federation.rounds.boundary.models import (
    RoundOpenRequest,
)
from scripts.experiments.federated_simulation.evaluation import (
    build_training_examples,
)
from scripts.experiments.federated_simulation.methods.base import (
    FederatedSslMethodDescriptor,
)
from scripts.experiments.federated_simulation.models import (
    FederatedTrainingTaskConfig,
)
from scripts.experiments.federated_simulation.task_config import (
    build_round_open_request,
)
from scripts.labeled_query_rows import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.training.shared_adapter_state import (
    SharedAdapterState,
)
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter

from .registry import register_federated_ssl_method

FEDAVG_PSEUDO_LABEL_DESCRIPTOR = FederatedSslMethodDescriptor(
    name="fedavg_pseudo_label",
    implementation_status="active_runtime",
    client_trainer_name="local_training_service",
    pseudo_labeler_name="agent_pseudo_label_selection",
    view_generator_name="training_example_backend",
    server_aggregator_name="round_runtime_aggregation_backend",
    round_policy_name="round_active_pair_only",
    requires_custom_client_runtime=False,
    requires_custom_server_runtime=False,
)


@dataclass(frozen=True, slots=True)
class FedAvgPseudoLabelRuntime:
    """기존 FedAvg + pseudo-label self-training 실행 조합."""

    descriptor: FederatedSslMethodDescriptor = FEDAVG_PSEUDO_LABEL_DESCRIPTOR

    def build_round_open_request(
        self,
        *,
        active_manifest: ModelManifest,
        round_id: str,
        training_task_config: FederatedTrainingTaskConfig,
    ) -> RoundOpenRequest:
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
        scoring_service: ScoringService,
        objective_config: TrainingObjectiveConfig,
    ) -> tuple[EmbeddedTrainingExample, ...]:
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
    ) -> LocalTrainingService:
        return LocalTrainingService(
            repository=TrainingArtifactRepository(state_root=client_state_root)
        )


@register_federated_ssl_method(
    "fedavg_pseudo_label",
    descriptor=FEDAVG_PSEUDO_LABEL_DESCRIPTOR,
)
def build_fedavg_pseudo_label_runtime() -> FedAvgPseudoLabelRuntime:
    """FedAvg pseudo-label baseline runtime을 생성한다."""

    return FedAvgPseudoLabelRuntime()

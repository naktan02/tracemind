"""FL simulation Query SSL LoRA-classifier local trainer adapter."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.services.training.execution.query_ssl_local_training_service import (
    QuerySslLocalTrainingService,
    QuerySslLoraLocalTrainingRequest,
)
from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)
from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
    materialize_base_lora_classifier_state,
)
from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
    build_lora_classifier_training_backend_config,
)
from methods.adaptation.lora_classifier.training.query_ssl_local_training import (
    QuerySslLoraClientTrainingResult,
)
from methods.adaptation.lora_classifier.training_backend import (
    LoraClassifierTrainingBackend,
)
from methods.federated.aggregation.base import FederatedAggregationContext
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedLocalTrainerRuntimeConfig,
    FederatedQuerySslObjectiveConfig,
)
from scripts.runtime_adapters.federated_agent.lora_classifier_artifacts import (
    SimulationQuerySslLoraDeltaMaterializer,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierState,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask


def run_query_ssl_lora_classifier_local_training(
    *,
    client_id: str,
    seed: int,
    output_dir: Path,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    active_adapter_state: object,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    query_ssl_config: FederatedQuerySslObjectiveConfig,
    trainer_runtime_config: FederatedLocalTrainerRuntimeConfig,
    lora_config: LoraClassifierTrainingBackendConfig | None = None,
    created_at: datetime | None = None,
) -> QuerySslLoraClientTrainingResult:
    """simulation runtime state를 method-owned Query SSL LoRA core에 연결한다."""

    if not isinstance(active_adapter_state, LoraClassifierState):
        raise ValueError(
            "Query SSL LoRA local training requires active LoraClassifierState."
        )
    effective_created_at = created_at or datetime.now(tz=timezone.utc)
    base_parameters = _load_base_parameters(
        active_adapter_state=active_adapter_state,
        output_dir=output_dir,
        aggregated_at=effective_created_at,
    )
    service = QuerySslLocalTrainingService(
        repository=TrainingArtifactRepository(
            state_root=output_dir / "agents" / client_id
        ),
        backend=LoraClassifierTrainingBackend(
            config=lora_config
            or build_lora_classifier_training_backend_config(
                training_task.objective_config
            )
        ),
    )
    return service.run_lora(
        QuerySslLoraLocalTrainingRequest(
            client_id=client_id,
            seed=seed,
            labeled_rows=labeled_rows,
            unlabeled_rows=unlabeled_rows,
            labels=tuple(str(label) for label in active_adapter_state.label_schema),
            base_parameters=base_parameters,
            training_task=training_task,
            model_manifest=model_manifest,
            query_ssl_config=query_ssl_config,
            trainer_runtime_config=trainer_runtime_config,
            created_at=effective_created_at,
            delta_materializer=SimulationQuerySslLoraDeltaMaterializer(
                output_dir=output_dir
            ),
        )
    )


def _load_base_parameters(
    *,
    active_adapter_state: LoraClassifierState,
    output_dir: Path,
    aggregated_at: datetime,
) -> LoraClassifierMaterializedState:
    return materialize_base_lora_classifier_state(
        base_state=active_adapter_state,
        context=FederatedAggregationContext(
            next_model_revision=active_adapter_state.model_revision,
            aggregated_at=aggregated_at,
            artifact_loader=AggregationArtifactStore(
                state_root=output_dir / "main_server" / "aggregation_artifacts"
            ),
        ),
    )

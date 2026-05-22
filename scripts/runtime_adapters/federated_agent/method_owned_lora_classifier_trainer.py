"""FL simulation method-owned LoRA-classifier local trainer adapter."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
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
from methods.federated.aggregation.base import FederatedAggregationContext
from methods.federated_ssl.lora_classifier_training import (
    run_method_owned_lora_classifier_training_core,
)
from methods.federated_ssl.peer_context import FederatedSslPeerContext
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedLocalTrainerRuntimeConfig,
    FederatedSslMethodConfig,
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


def run_method_owned_lora_classifier_local_training(
    *,
    client_id: str,
    seed: int,
    output_dir: Path,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    active_adapter_state: object,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    ssl_method_config: FederatedSslMethodConfig,
    strong_view_policy: str,
    unlabeled_batch_size: int | None,
    trainer_runtime_config: FederatedLocalTrainerRuntimeConfig,
    peer_context: FederatedSslPeerContext | None = None,
    lora_config: LoraClassifierTrainingBackendConfig | None = None,
    created_at: datetime | None = None,
) -> QuerySslLoraClientTrainingResult:
    """simulation runtime state를 선택된 method-owned LoRA core에 연결한다."""

    if not isinstance(active_adapter_state, LoraClassifierState):
        raise ValueError(
            "Method-owned LoRA local training requires active LoraClassifierState."
        )
    effective_created_at = created_at or datetime.now(tz=timezone.utc)
    base_parameters = _load_base_parameters(
        active_adapter_state=active_adapter_state,
        output_dir=output_dir,
        aggregated_at=effective_created_at,
    )
    result = run_method_owned_lora_classifier_training_core(
        client_id=client_id,
        seed=seed,
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        labels=tuple(str(label) for label in active_adapter_state.label_schema),
        base_parameters=base_parameters,
        training_task=training_task,
        model_manifest=model_manifest,
        ssl_method_config=ssl_method_config,
        peer_context=peer_context,
        strong_view_policy=strong_view_policy,
        unlabeled_batch_size=unlabeled_batch_size,
        lora_config=lora_config
        or build_lora_classifier_training_backend_config(
            training_task.objective_config
        ),
        trainer_runtime_config=trainer_runtime_config,
        created_at=effective_created_at,
        delta_materializer=SimulationQuerySslLoraDeltaMaterializer(
            output_dir=output_dir
        ),
    )
    TrainingArtifactRepository(
        state_root=output_dir / "agents" / client_id
    ).save_shared_adapter_update(
        result.update_envelope.update_id,
        result.update_payload,
    )
    return result


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

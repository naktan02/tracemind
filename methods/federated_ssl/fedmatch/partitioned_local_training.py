"""FedMatch partitioned local training runtime entrypoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from methods.adaptation.lora_classifier.federated_ssl import (
    partitioned_objective_training,
)
from methods.adaptation.text_classifier.peft_encoder.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.text_classifier.peft_encoder.federated_ssl.partitioned import (
    training_loop,
)
from methods.adaptation.text_classifier.peft_encoder.training import (
    query_ssl_local_training as qssl_training,
)
from methods.adaptation.text_classifier.peft_encoder.update.materialization import (
    LoraClassifierMaterializedState,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from methods.federated_ssl.peer_context import FederatedSslPeerContext
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask

LoraClassifierTrainerRuntimeConfig = qssl_training.LoraClassifierTrainerRuntimeConfig
QuerySslLoraClientTrainingResult = qssl_training.QuerySslLoraClientTrainingResult
QuerySslLoraDeltaMaterializer = qssl_training.QuerySslLoraDeltaMaterializer
QuerySslLoraObjectiveRuntimeConfig = qssl_training.QuerySslLoraObjectiveRuntimeConfig
PartitionedMethodLocalTrainingConfig = (
    partitioned_objective_training.PartitionedMethodLocalTrainingConfig
)


def run_method_owned_lora_classifier_training_core(
    *,
    client_id: str,
    seed: int,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    diagnostic_unlabeled_rows: Sequence[LabeledQueryRow] | None = None,
    labels: Sequence[str],
    base_parameters: LoraClassifierMaterializedState,
    base_partition_parameters: Mapping[str, LoraClassifierMaterializedState]
    | None = None,
    previous_client_partition_parameters: Mapping[str, LoraClassifierMaterializedState]
    | None = None,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    ssl_method_config: PartitionedMethodLocalTrainingConfig,
    local_ssl_policy_name: str,
    query_ssl_config: QuerySslLoraObjectiveRuntimeConfig | None,
    strong_view_policy: str,
    unlabeled_batch_size: int | None,
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: LoraClassifierTrainerRuntimeConfig,
    created_at: datetime,
    delta_materializer: QuerySslLoraDeltaMaterializer,
    peer_context: FederatedSslPeerContext | None = None,
    helper_weak_probability_provider: (
        training_loop.HelperWeakProbabilityProvider | None
    ) = None,
    peer_probe_rows: Sequence[LabeledQueryRow] | None = None,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    timing_recorder: TimingRecorder | None = None,
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> QuerySslLoraClientTrainingResult:
    """FedMatch descriptor가 호출하는 PEFT text classifier partitioned runtime."""

    run_partitioned_training = (
        partitioned_objective_training.run_method_owned_lora_classifier_training_core
    )
    return run_partitioned_training(
        client_id=client_id,
        seed=seed,
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        diagnostic_unlabeled_rows=diagnostic_unlabeled_rows,
        labels=labels,
        base_parameters=base_parameters,
        base_partition_parameters=base_partition_parameters,
        previous_client_partition_parameters=previous_client_partition_parameters,
        training_task=training_task,
        model_manifest=model_manifest,
        ssl_method_config=ssl_method_config,
        local_ssl_policy_name=local_ssl_policy_name,
        query_ssl_config=query_ssl_config,
        strong_view_policy=strong_view_policy,
        unlabeled_batch_size=unlabeled_batch_size,
        lora_config=lora_config,
        trainer_runtime_config=trainer_runtime_config,
        created_at=created_at,
        delta_materializer=delta_materializer,
        peer_context=peer_context,
        helper_weak_probability_provider=helper_weak_probability_provider,
        peer_probe_rows=peer_probe_rows,
        runtime_resource_cache=runtime_resource_cache,
        timing_recorder=timing_recorder,
        initial_query_ssl_algorithm_state=initial_query_ssl_algorithm_state,
    )

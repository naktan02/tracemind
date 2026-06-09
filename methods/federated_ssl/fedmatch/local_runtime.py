"""FedMatch method-owned local runtime entrypoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.federated_ssl import (
    partitioned_objective_training,
)
from methods.adaptation.peft_text_encoder.federated_ssl.method_training_surface import (
    FsslPeftEncoderMethodTrainingRequest,
)
from methods.adaptation.peft_text_encoder.federated_ssl.partitioned import (
    training_loop,
)
from methods.adaptation.peft_text_encoder.training import (
    query_ssl_local_training as qssl_training,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from methods.federated_ssl.fedmatch.partitioning import (
    build_fedmatch_partitioned_runtime_plan,
)
from methods.federated_ssl.hooks.peer_context import FederatedSslPeerContext
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask

PeftEncoderTrainerRuntimeConfig = qssl_training.PeftEncoderTrainerRuntimeConfig
QuerySslPeftEncoderClientTrainingResult = (
    qssl_training.QuerySslPeftEncoderClientTrainingResult
)
QuerySslPeftEncoderDeltaMaterializer = (
    qssl_training.QuerySslPeftEncoderDeltaMaterializer
)
QuerySslPeftEncoderObjectiveRuntimeConfig = (
    qssl_training.QuerySslPeftEncoderObjectiveRuntimeConfig
)
PartitionedMethodLocalTrainingConfig = (
    partitioned_objective_training.PartitionedMethodLocalTrainingConfig
)


def run_method_owned_peft_encoder_training_core(
    *,
    client_id: str,
    seed: int,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    diagnostic_unlabeled_rows: Sequence[LabeledQueryRow] | None = None,
    labels: Sequence[str],
    base_parameters: PeftEncoderMaterializedState,
    base_partition_parameters: Mapping[str, PeftEncoderMaterializedState] | None = None,
    previous_client_partition_parameters: Mapping[str, PeftEncoderMaterializedState]
    | None = None,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    ssl_method_config: PartitionedMethodLocalTrainingConfig,
    local_ssl_policy_name: str,
    query_ssl_config: QuerySslPeftEncoderObjectiveRuntimeConfig | None,
    strong_view_policy: str,
    unlabeled_batch_size: int | None,
    peft_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    created_at: datetime,
    delta_materializer: QuerySslPeftEncoderDeltaMaterializer,
    peer_context: FederatedSslPeerContext | None = None,
    helper_weak_probability_provider: (
        training_loop.HelperWeakProbabilityProvider | None
    ) = None,
    peer_probe_rows: Sequence[LabeledQueryRow] | None = None,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    timing_recorder: TimingRecorder | None = None,
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> QuerySslPeftEncoderClientTrainingResult:
    """호환용 kwargs entrypoint. Descriptor는 request entrypoint를 사용한다."""

    return run_method_owned_peft_encoder_training_request(
        FsslPeftEncoderMethodTrainingRequest(
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
            peft_config=peft_config,
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
    )


def run_method_owned_peft_encoder_training_request(
    request: FsslPeftEncoderMethodTrainingRequest,
) -> QuerySslPeftEncoderClientTrainingResult:
    """FedMatch descriptor가 호출하는 PEFT text encoder local runtime."""

    runtime_plan = build_fedmatch_partitioned_runtime_plan(
        scenario_name=request.ssl_method_config.scenario,
        effective_parameters=request.ssl_method_config.effective_parameters,
    )
    return partitioned_objective_training.run_partitioned_peft_encoder_training_core(
        client_id=request.client_id,
        seed=request.seed,
        labeled_rows=request.labeled_rows,
        unlabeled_rows=request.unlabeled_rows,
        diagnostic_unlabeled_rows=request.diagnostic_unlabeled_rows,
        labels=request.labels,
        base_parameters=request.base_parameters,
        base_partition_parameters=request.base_partition_parameters,
        previous_client_partition_parameters=(
            request.previous_client_partition_parameters
        ),
        training_task=request.training_task,
        model_manifest=request.model_manifest,
        ssl_method_config=request.ssl_method_config,
        partitioned_runtime_plan=runtime_plan,
        local_ssl_policy_name=request.local_ssl_policy_name,
        query_ssl_config=request.query_ssl_config,
        strong_view_policy=request.strong_view_policy,
        unlabeled_batch_size=request.unlabeled_batch_size,
        peft_config=request.peft_config,
        trainer_runtime_config=request.trainer_runtime_config,
        created_at=request.created_at,
        delta_materializer=request.delta_materializer,
        peer_context=request.peer_context,
        helper_weak_probability_provider=request.helper_weak_probability_provider,
        peer_probe_rows=request.peer_probe_rows,
        runtime_resource_cache=request.runtime_resource_cache,
        timing_recorder=request.timing_recorder,
        initial_query_ssl_algorithm_state=request.initial_query_ssl_algorithm_state,
    )

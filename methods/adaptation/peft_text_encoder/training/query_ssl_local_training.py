"""Query SSL PEFT text encoder/head local training orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask

from .local_training_surface import QuerySslPeftEncoderUpdateRequest
from .query_ssl_federated_update import (
    QuerySslPeftEncoderClientTrainingResult,
    QuerySslPeftEncoderDeltaMaterialization,  # noqa: F401
    QuerySslPeftEncoderDeltaMaterializer,
    build_query_ssl_peft_encoder_client_update,
)
from .query_ssl_training_session import (
    PeftEncoderTrainerRuntimeConfig,
    QuerySslPeftEncoderLocalSslResult,  # noqa: F401
    QuerySslPeftEncoderObjectiveRuntimeConfig,
    _build_bounded_label_balanced_selection_rows,  # noqa: F401
    run_query_ssl_peft_encoder_local_ssl,
)


def run_query_ssl_peft_encoder_training_core(
    *,
    client_id: str,
    seed: int,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    diagnostic_unlabeled_rows: Sequence[LabeledQueryRow] | None = None,
    labels: Sequence[str],
    base_parameters: PeftEncoderMaterializedState,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    query_ssl_config: QuerySslPeftEncoderObjectiveRuntimeConfig,
    peft_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    created_at: datetime,
    delta_materializer: QuerySslPeftEncoderDeltaMaterializer,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    timing_recorder: TimingRecorder | None = None,
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> QuerySslPeftEncoderClientTrainingResult:
    """client-local raw text/views로 Query SSL PEFT encoder update를 생성한다."""

    local_ssl_result = run_query_ssl_peft_encoder_local_ssl(
        seed=seed,
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        diagnostic_unlabeled_rows=diagnostic_unlabeled_rows,
        labels=labels,
        base_parameters=base_parameters,
        training_task=training_task,
        query_ssl_config=query_ssl_config,
        peft_config=peft_config,
        trainer_runtime_config=trainer_runtime_config,
        runtime_resource_cache=runtime_resource_cache,
        timing_recorder=timing_recorder,
        initial_query_ssl_algorithm_state=initial_query_ssl_algorithm_state,
    )
    return build_query_ssl_peft_encoder_client_update(
        client_id=client_id,
        training_task=training_task,
        model_manifest=model_manifest,
        peft_config=peft_config,
        created_at=created_at,
        base_parameters=base_parameters,
        delta_materializer=delta_materializer,
        local_ssl_result=local_ssl_result,
        timing_recorder=timing_recorder,
    )


def run_query_ssl_peft_encoder_update(
    request: QuerySslPeftEncoderUpdateRequest,
) -> QuerySslPeftEncoderClientTrainingResult:
    """공통 update request surface로 Query SSL PEFT encoder update를 생성한다."""

    session = request.local_session
    return run_query_ssl_peft_encoder_training_core(
        client_id=request.client_id,
        seed=session.seed,
        labeled_rows=session.labeled_rows,
        unlabeled_rows=session.unlabeled_rows,
        diagnostic_unlabeled_rows=session.diagnostic_unlabeled_rows,
        labels=session.labels,
        base_parameters=session.base_parameters,
        training_task=session.training_task,
        model_manifest=request.model_manifest,
        query_ssl_config=session.query_ssl_config,
        peft_config=session.peft_config,
        trainer_runtime_config=session.trainer_runtime_config,
        created_at=request.created_at,
        delta_materializer=request.delta_materializer,
        runtime_resource_cache=session.runtime_resource_cache,
        timing_recorder=session.timing_recorder,
        initial_query_ssl_algorithm_state=session.initial_query_ssl_algorithm_state,
    )

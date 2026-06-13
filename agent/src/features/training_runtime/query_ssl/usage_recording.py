"""Query SSL source usage ledger recording."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from agent.src.features.training_runtime.query_ssl.task_identity import (
    optional_method_name,
)
from agent.src.features.training_runtime.query_ssl_peft.local_training_service import (
    QuerySslPeftEncoderClientTrainingResult,
)
from agent.src.features.training_runtime.storage.training_usage_ledger_repository import (  # noqa: E501
    TRAINING_USAGE_ROLE_LABELED_ANCHOR,
    TRAINING_USAGE_ROLE_UNLABELED_GENERATED_VIEW,
    TRAINING_USAGE_STAGE_QUERY_SSL_INPUT,
    TRAINING_USAGE_STATUS_UPLOADED,
    TrainingUsageLedgerRepository,
    TrainingUsageRowRecord,
    TrainingUsageRunRecord,
)
from methods.ssl.runtime.objective_config import QuerySslObjectiveRuntimeConfig
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.training_contracts import TrainingTask


def record_query_ssl_training_usage(
    *,
    repository: TrainingUsageLedgerRepository,
    training_task: TrainingTask,
    agent_id: str | None,
    query_ssl_config: QuerySslObjectiveRuntimeConfig,
    local_result: QuerySslPeftEncoderClientTrainingResult,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    recorded_at: datetime,
) -> None:
    """Query SSL local update에 사용한 source row를 ledger에 기록한다."""

    repository.save_run(
        _build_usage_run_record(
            training_task=training_task,
            agent_id=agent_id,
            query_ssl_config=query_ssl_config,
            local_result=local_result,
            recorded_at=recorded_at,
        ),
        rows=_build_usage_row_records(
            update_id=local_result.update_envelope.update_id,
            training_task=training_task,
            labeled_rows=labeled_rows,
            unlabeled_rows=unlabeled_rows,
            recorded_at=recorded_at,
        ),
    )


def _build_usage_run_record(
    *,
    training_task: TrainingTask,
    agent_id: str | None,
    query_ssl_config: QuerySslObjectiveRuntimeConfig,
    local_result: QuerySslPeftEncoderClientTrainingResult,
    recorded_at: datetime,
) -> TrainingUsageRunRecord:
    fssl_method = optional_method_name(training_task.fssl_method)
    effective_method_family = (
        "federated_ssl" if fssl_method is not None else "query_ssl"
    )
    return TrainingUsageRunRecord(
        update_id=local_result.update_envelope.update_id,
        round_id=training_task.round_id,
        task_id=training_task.task_id,
        recorded_at=recorded_at,
        agent_id=agent_id,
        model_id=training_task.model_id,
        model_revision=training_task.model_revision,
        objective_method_name=fssl_method or query_ssl_config.method_name,
        objective_algorithm_name=(
            effective_method_family
            if fssl_method is not None
            else query_ssl_config.algorithm_name
        ),
        status=TRAINING_USAGE_STATUS_UPLOADED,
        candidate_count=local_result.candidate_count,
        accepted_count=local_result.accepted_count,
        metadata={
            "effective_method_family": effective_method_family,
            "fssl_method": fssl_method,
            "query_ssl_method_name": query_ssl_config.method_name,
            "query_ssl_algorithm_name": query_ssl_config.algorithm_name,
            "training_scope": str(training_task.training_scope),
            "local_epochs": training_task.local_epochs,
            "max_steps": training_task.max_steps,
            "batch_size": training_task.batch_size,
            "learning_rate": training_task.learning_rate,
            "selection_max_examples": training_task.selection_policy.max_examples,
        },
    )


def _build_usage_row_records(
    *,
    update_id: str,
    training_task: TrainingTask,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    recorded_at: datetime,
) -> tuple[TrainingUsageRowRecord, ...]:
    records: list[TrainingUsageRowRecord] = []
    for row in labeled_rows:
        records.append(
            TrainingUsageRowRecord(
                update_id=update_id,
                source_id=str(row["query_id"]),
                role=TRAINING_USAGE_ROLE_LABELED_ANCHOR,
                round_id=training_task.round_id,
                task_id=training_task.task_id,
                recorded_at=recorded_at,
                source_kind="analysis_event",
                stage=TRAINING_USAGE_STAGE_QUERY_SSL_INPUT,
                label=str(row["mapped_label_4"]),
                metadata={
                    "annotation_source": str(row["annotation_source"]),
                    "raw_label_scheme": str(row["raw_label_scheme"]),
                },
            )
        )
    for row in unlabeled_rows:
        records.append(
            TrainingUsageRowRecord(
                update_id=update_id,
                source_id=str(row["query_id"]),
                role=TRAINING_USAGE_ROLE_UNLABELED_GENERATED_VIEW,
                round_id=training_task.round_id,
                task_id=training_task.task_id,
                recorded_at=recorded_at,
                source_kind="captured_text_generated_view",
                stage=TRAINING_USAGE_STAGE_QUERY_SSL_INPUT,
                label=None,
                metadata={
                    "annotation_source": str(row["annotation_source"]),
                    "weak_translated": bool(row.get("weak_translated_text")),
                    "strong_translated": bool(row.get("strong_translated_text")),
                },
            )
        )
    return tuple(records)

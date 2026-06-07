"""Build federated Query SSL updates from a local PEFT training session."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.adaptation.peft_text_encoder.update.partitioned_delta import (
    PeftEncoderPartitionDelta,
)
from methods.adaptation.peft_text_encoder.update.query_ssl_update import (
    build_query_ssl_peft_encoder_update_payload,
)
from methods.adaptation.query_text_views.local_training_budget import (
    QuerySslLocalStepPlan,
)
from methods.common.timing import TimingRecorder, timing_mapping
from methods.evaluation.pseudo_label_quality import PseudoLabelQualitySummary
from methods.federated_ssl.hooks.peer_context import FederatedSslPeerClientSnapshot
from methods.ssl.state import export_query_ssl_algorithm_state
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    PeftClassifierDelta,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingTask,
    TrainingUpdateEnvelope,
    make_training_update_envelope,
)

from .delta_extraction import extract_peft_encoder_parameter_deltas
from .query_ssl_training_session import QuerySslPeftEncoderLocalSslResult


@dataclass(frozen=True, slots=True)
class QuerySslPeftEncoderDeltaMaterialization:
    """PEFT encoder/head delta가 update payload에 담기는 방식."""

    delta_format: str
    peft_adapter_delta_artifact_ref: str | None
    classifier_head_delta_artifact_ref: str | None
    include_inline_deltas: bool
    partitioned_deltas_artifact_ref: str | None = None


class QuerySslPeftEncoderDeltaMaterializer(Protocol):
    """runtime별 delta artifact 저장소 bridge."""

    def prepare(
        self,
        *,
        update_id: str,
        training_task: TrainingTask,
        client_id: str,
        delta_format: str,
        artifact_ref_prefix: str,
        peft_parameter_deltas: Mapping[str, Sequence[float]],
        classifier_head_weight_deltas: Mapping[str, Sequence[float]],
        classifier_head_bias_deltas: Mapping[str, float],
        partitioned_deltas: Mapping[str, PeftEncoderPartitionDelta] | None = None,
        materialize_primary_deltas: bool = True,
    ) -> QuerySslPeftEncoderDeltaMaterialization:
        """delta 저장 방식을 결정하고 artifact ref를 반환한다."""


@dataclass(frozen=True, slots=True)
class QuerySslPeftEncoderClientTrainingResult:
    """FL round loop가 서버 제출과 client summary에 쓰는 local training 결과."""

    update_envelope: TrainingUpdateEnvelope
    update_payload: PeftClassifierDelta
    candidate_count: int
    accepted_count: int
    local_step_plan: QuerySslLocalStepPlan
    client_metrics: Mapping[str, float]
    pseudo_label_quality: PseudoLabelQualitySummary = field(
        default_factory=PseudoLabelQualitySummary.empty
    )
    peer_client_snapshot: FederatedSslPeerClientSnapshot | None = None
    client_partition_parameters: Mapping[str, PeftEncoderMaterializedState] = field(
        default_factory=dict
    )
    query_ssl_algorithm_state: Mapping[str, Any] = field(default_factory=dict)
    timing_breakdown: Mapping[str, float] = field(default_factory=dict)


def build_query_ssl_peft_encoder_client_update(
    *,
    client_id: str,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    peft_config: PeftEncoderTrainingBackendConfig,
    created_at: datetime,
    base_parameters: PeftEncoderMaterializedState,
    delta_materializer: QuerySslPeftEncoderDeltaMaterializer,
    local_ssl_result: QuerySslPeftEncoderLocalSslResult,
    timing_recorder: TimingRecorder | None = None,
) -> QuerySslPeftEncoderClientTrainingResult:
    """학습된 local SSL 결과를 federated client update payload로 변환한다."""

    with _measure(timing_recorder, "core_delta_extract_seconds"):
        peft_parameter_deltas, head_weight_deltas, head_bias_deltas = (
            extract_peft_encoder_parameter_deltas(
                model=local_ssl_result.model,
                base_parameters=base_parameters,
                labels=local_ssl_result.effective_labels,
            )
        )
    update_id = f"update_{training_task.round_id}_{client_id}_{uuid4().hex[:12]}"
    with _measure(timing_recorder, "core_delta_materialization_seconds"):
        delta_materialization = delta_materializer.prepare(
            update_id=update_id,
            training_task=training_task,
            client_id=client_id,
            delta_format=peft_config.delta_format,
            artifact_ref_prefix=peft_config.artifact_ref_prefix,
            peft_parameter_deltas=peft_parameter_deltas,
            classifier_head_weight_deltas=head_weight_deltas,
            classifier_head_bias_deltas=head_bias_deltas,
        )
    with _measure(timing_recorder, "core_update_payload_build_seconds"):
        update_build_result = build_query_ssl_peft_encoder_update_payload(
            training_task=training_task,
            model_manifest=model_manifest,
            peft_config=peft_config,
            labels=local_ssl_result.effective_labels,
            labeled_rows=local_ssl_result.effective_labeled_rows,
            unlabeled_rows=local_ssl_result.effective_unlabeled_rows,
            step_plan=local_ssl_result.local_step_plan,
            history_record=(
                local_ssl_result.history[-1] if local_ssl_result.history else {}
            ),
            peft_parameter_deltas=peft_parameter_deltas,
            classifier_head_weight_deltas=head_weight_deltas,
            classifier_head_bias_deltas=head_bias_deltas,
            created_at=created_at,
            delta_format=delta_materialization.delta_format,
            peft_adapter_delta_artifact_ref=delta_materialization.peft_adapter_delta_artifact_ref,
            classifier_head_delta_artifact_ref=(
                delta_materialization.classifier_head_delta_artifact_ref
            ),
            include_inline_deltas=delta_materialization.include_inline_deltas,
        )
    update_payload = update_build_result.update_payload
    client_metrics = {
        **dict(update_build_result.client_metrics),
        "query_ssl_selection_labeled_count": float(
            len(local_ssl_result.selection_rows)
        ),
        **dict(local_ssl_result.diagnostic_client_metrics),
    }
    update_envelope = make_training_update_envelope(
        update_id=update_id,
        round_id=training_task.round_id,
        task_id=training_task.task_id,
        model_id=model_manifest.model_id,
        base_model_revision=model_manifest.model_revision,
        training_scope=training_task.training_scope,
        payload_ref=f"client-submission::{update_id}",
        payload_format=_payload_format_for_update(update_payload),
        example_count=update_payload.example_count,
        client_metrics=dict(client_metrics),
        created_at=created_at,
    )
    return QuerySslPeftEncoderClientTrainingResult(
        update_envelope=update_envelope,
        update_payload=update_payload,
        candidate_count=len(local_ssl_result.effective_unlabeled_rows),
        accepted_count=update_build_result.accepted_unlabeled_count,
        local_step_plan=local_ssl_result.local_step_plan,
        client_metrics=client_metrics,
        pseudo_label_quality=local_ssl_result.pseudo_label_quality,
        query_ssl_algorithm_state=dict(
            export_query_ssl_algorithm_state(local_ssl_result.algorithm)
        ),
        timing_breakdown=timing_mapping(timing_recorder),
    )


def _payload_format_for_update(update_payload: PeftClassifierDelta) -> str:
    return PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT


def _measure(timing_recorder: TimingRecorder | None, key: str) -> Any:
    if timing_recorder is None:
        return nullcontext()
    return timing_recorder.measure(key)

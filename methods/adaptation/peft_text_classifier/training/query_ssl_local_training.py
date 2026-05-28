"""Query SSL PEFT-backed classifier local training core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from methods.adaptation.peft_text_classifier.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_classifier.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.adaptation.peft_text_classifier.update.partitioned_delta import (
    PeftEncoderPartitionDelta,
)
from methods.adaptation.peft_text_classifier.update.query_ssl_update import (
    build_query_ssl_peft_encoder_update_payload,
)
from methods.adaptation.query_text_views.data import (
    build_dataloader,
)
from methods.adaptation.query_text_views.local_training_budget import (
    QuerySslLocalStepPlan,
    build_query_ssl_local_step_plan,
)
from methods.adaptation.query_text_views.query_ssl_views import (
    build_query_ssl_unlabeled_dataloader,
)
from methods.adaptation.query_text_views.tokenization import (
    TextTokenizationCache,
    resolve_text_tokenization_cache,
)
from methods.adaptation.query_text_views.view_rows import (
    validate_query_ssl_unlabeled_views,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder, timing_mapping
from methods.evaluation.pseudo_label_quality import PseudoLabelQualitySummary
from methods.federated_ssl.peer_context import FederatedSslPeerClientSnapshot
from methods.ssl.registry import resolve_query_ssl_algorithm_descriptor
from methods.ssl.state import export_query_ssl_algorithm_state
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    PeftClassifierDelta,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingTask,
    TrainingUpdateEnvelope,
    make_training_update_envelope,
)

from .delta_extraction import (
    extract_peft_encoder_parameter_deltas,
    load_peft_encoder_base_parameters_into_model,
)
from .loops import set_seed, train_query_ssl_classifier
from .modeling import (
    PeftEncoderTextClassifier,
    build_peft_encoder_text_classifier_from_config,
)
from .pseudo_label_diagnostics import (
    build_final_snapshot_pseudo_label_quality,
    resolve_fixed_pseudo_label_diagnostic_threshold,
    tokenization_cache_namespace,
)


class QuerySslPeftEncoderObjectiveRuntimeConfig(Protocol):
    """Query SSL PEFT encoder local core가 필요한 objective config surface."""

    algorithm_name: str
    parameters: Mapping[str, object]
    strong_view_policy: str
    unlabeled_batch_size: int | None


class PeftEncoderTrainerRuntimeConfig(Protocol):
    """PEFT encoder classifier 모델 로딩/학습 core가 필요한 runtime config surface."""

    device: str
    classifier_dropout: float
    cache_dir: str | None
    local_files_only: bool
    trust_remote_code: bool


@dataclass(frozen=True, slots=True)
class QuerySslPeftEncoderDeltaMaterialization:
    """PEFT encoder/head delta가 update payload에 담기는 방식."""

    delta_format: str
    lora_delta_artifact_ref: str | None
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
        lora_parameter_deltas: Mapping[str, Sequence[float]],
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
    lora_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    created_at: datetime,
    delta_materializer: QuerySslPeftEncoderDeltaMaterializer,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    timing_recorder: TimingRecorder | None = None,
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> QuerySslPeftEncoderClientTrainingResult:
    """client-local raw text/views로 Query SSL PEFT encoder update를 생성한다."""

    effective_labeled_rows = list(labeled_rows)
    effective_unlabeled_rows = list(unlabeled_rows)
    if not effective_labeled_rows:
        raise ValueError("Query SSL PEFT encoder local training requires labeled_rows.")
    if not effective_unlabeled_rows:
        raise ValueError(
            "Query SSL PEFT encoder local training requires unlabeled_rows."
        )

    descriptor = resolve_query_ssl_algorithm_descriptor(query_ssl_config.algorithm_name)
    validate_query_ssl_unlabeled_views(
        rows=effective_unlabeled_rows,
        view_builder_name=descriptor.required_views.view_builder_name,
        algorithm_name=descriptor.algorithm_name,
    )
    algorithm = descriptor.build_algorithm(query_ssl_config.parameters)
    effective_labels = tuple(str(label) for label in labels)
    if not effective_labels:
        raise ValueError("PEFT encoder classifier label schema must not be empty.")
    _validate_labeled_rows_have_known_labels(
        rows=effective_labeled_rows,
        labels=effective_labels,
    )
    tokenization_cache = resolve_text_tokenization_cache(runtime_resource_cache)
    tokenization_cache_namespace_value = tokenization_cache_namespace(lora_config)

    with _measure(timing_recorder, "core_model_prepare_seconds"):
        with _measure(timing_recorder, "core_seed_seconds"):
            set_seed(int(seed))
        with _measure(timing_recorder, "core_model_build_seconds"):
            model, tokenizer = _build_peft_encoder_model(
                labels=effective_labels,
                lora_config=lora_config,
                trainer_runtime_config=trainer_runtime_config,
                runtime_resource_cache=runtime_resource_cache,
            )
        with _measure(timing_recorder, "core_base_parameter_load_seconds"):
            load_peft_encoder_base_parameters_into_model(
                model=model,
                labels=effective_labels,
                base_parameters=base_parameters,
                device=trainer_runtime_config.device,
            )

    with _measure(timing_recorder, "core_dataloader_prepare_seconds"):
        label_to_index = {label: index for index, label in enumerate(effective_labels)}
        train_loader = build_dataloader(
            rows=effective_labeled_rows,
            label_to_index=label_to_index,
            tokenizer=tokenizer,
            batch_size=int(training_task.batch_size),
            max_length=lora_config.max_length,
            task_prefix=lora_config.task_prefix,
            shuffle=True,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace_value,
        )
        selection_loader = build_dataloader(
            rows=effective_labeled_rows,
            label_to_index=label_to_index,
            tokenizer=tokenizer,
            batch_size=int(training_task.batch_size),
            max_length=lora_config.max_length,
            task_prefix=lora_config.task_prefix,
            shuffle=False,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace_value,
        )
        unlabeled_loader = _build_unlabeled_loader(
            rows=effective_unlabeled_rows,
            tokenizer=tokenizer,
            batch_size=query_ssl_config.unlabeled_batch_size
            or int(training_task.batch_size),
            max_length=lora_config.max_length,
            task_prefix=lora_config.task_prefix,
            strong_view_policy=query_ssl_config.strong_view_policy,
            view_builder_name=descriptor.required_views.view_builder_name,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace_value,
        )
        step_plan = build_query_ssl_local_step_plan(
            labeled_loader_steps=len(train_loader),
            unlabeled_loader_steps=len(unlabeled_loader),
            uses_labeled_batches=algorithm.uses_labeled_batches,
            local_epochs=int(training_task.local_epochs),
            max_steps=int(training_task.max_steps),
        )

    with _measure(timing_recorder, "core_training_loop_seconds"):
        model, history, _best_selection_report = train_query_ssl_classifier(
            model=model,
            train_loader=train_loader,
            unlabeled_loader=unlabeled_loader,
            selection_loader=selection_loader,
            categories=list(effective_labels),
            device=trainer_runtime_config.device,
            epochs=int(training_task.local_epochs),
            max_train_steps=step_plan.total_steps,
            learning_rate=float(training_task.learning_rate),
            classifier_learning_rate=float(training_task.learning_rate),
            weight_decay=0.0,
            proximal_mu=float(lora_config.proximal_mu),
            max_grad_norm=(
                0.0
                if training_task.gradient_clip_norm is None
                else float(training_task.gradient_clip_norm)
            ),
            log_every_steps=0,
            algorithm=algorithm,
            initial_query_ssl_algorithm_state=initial_query_ssl_algorithm_state,
        )
    diagnostic_threshold = resolve_fixed_pseudo_label_diagnostic_threshold(
        query_ssl_config.parameters
    )
    with _measure(timing_recorder, "core_pseudo_label_diagnostics_seconds"):
        pseudo_label_quality = build_final_snapshot_pseudo_label_quality(
            model=model,
            tokenizer=tokenizer,
            rows=(
                effective_unlabeled_rows
                if diagnostic_unlabeled_rows is None
                else list(diagnostic_unlabeled_rows)
            ),
            labels=effective_labels,
            lora_config=lora_config,
            acceptance_threshold=diagnostic_threshold.threshold,
            trainer_runtime_config=trainer_runtime_config,
            unlabeled_batch_size=query_ssl_config.unlabeled_batch_size or 1,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace_value,
        )

    with _measure(timing_recorder, "core_delta_extract_seconds"):
        lora_deltas, head_weight_deltas, head_bias_deltas = (
            extract_peft_encoder_parameter_deltas(
                model=model,
                base_parameters=base_parameters,
                labels=effective_labels,
            )
        )
    update_id = f"update_{training_task.round_id}_{client_id}_{uuid4().hex[:12]}"
    with _measure(timing_recorder, "core_delta_materialization_seconds"):
        delta_materialization = delta_materializer.prepare(
            update_id=update_id,
            training_task=training_task,
            client_id=client_id,
            delta_format=lora_config.delta_format,
            artifact_ref_prefix=lora_config.artifact_ref_prefix,
            lora_parameter_deltas=lora_deltas,
            classifier_head_weight_deltas=head_weight_deltas,
            classifier_head_bias_deltas=head_bias_deltas,
        )
    with _measure(timing_recorder, "core_update_payload_build_seconds"):
        update_build_result = build_query_ssl_peft_encoder_update_payload(
            training_task=training_task,
            model_manifest=model_manifest,
            lora_config=lora_config,
            labels=effective_labels,
            labeled_rows=effective_labeled_rows,
            unlabeled_rows=effective_unlabeled_rows,
            step_plan=step_plan,
            history_record=history[-1] if history else {},
            lora_parameter_deltas=lora_deltas,
            classifier_head_weight_deltas=head_weight_deltas,
            classifier_head_bias_deltas=head_bias_deltas,
            created_at=created_at,
            delta_format=delta_materialization.delta_format,
            lora_delta_artifact_ref=delta_materialization.lora_delta_artifact_ref,
            classifier_head_delta_artifact_ref=(
                delta_materialization.classifier_head_delta_artifact_ref
            ),
            include_inline_deltas=delta_materialization.include_inline_deltas,
        )
    update_payload = update_build_result.update_payload
    client_metrics = {
        **dict(update_build_result.client_metrics),
        **diagnostic_threshold.to_client_metrics(),
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
        candidate_count=len(effective_unlabeled_rows),
        accepted_count=update_build_result.accepted_unlabeled_count,
        local_step_plan=step_plan,
        client_metrics=client_metrics,
        pseudo_label_quality=pseudo_label_quality,
        query_ssl_algorithm_state=dict(export_query_ssl_algorithm_state(algorithm)),
        timing_breakdown=timing_mapping(timing_recorder),
    )


def _build_peft_encoder_model(
    *,
    labels: Sequence[str],
    lora_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None,
) -> tuple[PeftEncoderTextClassifier, Any]:
    return build_peft_encoder_text_classifier_from_config(
        labels=[str(label) for label in labels],
        lora_config=lora_config,
        runtime_config=trainer_runtime_config,
        runtime_resource_cache=runtime_resource_cache,
    )


def _measure(timing_recorder: TimingRecorder | None, key: str) -> Any:
    if timing_recorder is None:
        return nullcontext()
    return timing_recorder.measure(key)


def _payload_format_for_update(update_payload: PeftClassifierDelta) -> str:
    return PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT


def _build_unlabeled_loader(
    *,
    rows: Sequence[LabeledQueryRow],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    task_prefix: str,
    strong_view_policy: str,
    view_builder_name: str,
    tokenization_cache: TextTokenizationCache | None,
    tokenization_cache_namespace: str,
) -> Any:
    return build_query_ssl_unlabeled_dataloader(
        rows=rows,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        task_prefix=task_prefix,
        shuffle=True,
        view_builder_name=view_builder_name,
        strong_view_policy=strong_view_policy,
        tokenization_cache=tokenization_cache,
        tokenization_cache_namespace=tokenization_cache_namespace,
    )


def _validate_labeled_rows_have_known_labels(
    *,
    rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
) -> None:
    known_labels = {str(label) for label in labels}
    missing = sorted({str(row["mapped_label_4"]) for row in rows} - known_labels)
    if missing:
        raise ValueError(
            "Query SSL labeled_rows contain labels outside active label_schema: "
            f"{missing}."
        )

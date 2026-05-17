"""FL simulation Query SSL LoRA-classifier local trainer adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)
from methods.adaptation.lora.lora_adapter import resolve_target_modules
from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
    materialize_base_lora_classifier_state,
)
from methods.adaptation.lora_classifier.config import (
    LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
    LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED,
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.lora_classifier.delta_extraction import (
    extract_lora_classifier_parameter_deltas,
    load_lora_classifier_base_parameters_into_model,
)
from methods.adaptation.lora_classifier.modeling import (
    LoraTextClassifier,
    require_transformer_stack,
)
from methods.adaptation.lora_classifier.query_ssl_update import (
    build_query_ssl_lora_update_payload,
)
from methods.adaptation.lora_classifier.training import (
    set_seed,
    train_query_ssl_classifier,
)
from methods.adaptation.query_classifier_adaptation.data import (
    build_dataloader,
    build_multiview_dataloader,
    build_weak_dataloader,
)
from methods.adaptation.query_classifier_adaptation.local_training_budget import (
    QuerySslLocalStepPlan,
    build_query_ssl_local_step_plan,
)
from methods.adaptation.query_classifier_adaptation.view_rows import (
    USB_MULTIVIEW_BUILDER_NAME,
    USB_WEAK_BUILDER_NAME,
    validate_query_ssl_unlabeled_views,
)
from methods.federated.aggregation.base import FederatedAggregationContext
from methods.ssl.registry import resolve_query_ssl_algorithm_descriptor
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedLocalTrainerRuntimeConfig,
    FederatedQuerySslObjectiveConfig,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    LoraClassifierDelta,
    LoraClassifierState,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingTask,
    TrainingUpdateEnvelope,
    make_training_update_envelope,
)


@dataclass(frozen=True, slots=True)
class QuerySslLoraClientTrainingResult:
    """FL round loop가 서버 제출과 client summary에 쓰는 local training 결과."""

    update_envelope: TrainingUpdateEnvelope
    update_payload: LoraClassifierDelta
    candidate_count: int
    accepted_count: int
    local_step_plan: QuerySslLocalStepPlan
    client_metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class _DeltaMaterializationPlan:
    """simulation runtime이 LoRA/head delta를 update payload에 담는 방식."""

    delta_format: str
    lora_delta_artifact_ref: str | None
    classifier_head_delta_artifact_ref: str | None
    include_inline_deltas: bool


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
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: FederatedLocalTrainerRuntimeConfig,
    created_at: datetime | None = None,
) -> QuerySslLoraClientTrainingResult:
    """client-local raw text/views로 Query SSL LoRA update를 생성한다."""

    if not isinstance(active_adapter_state, LoraClassifierState):
        raise ValueError(
            "Query SSL LoRA local training requires active LoraClassifierState."
        )
    effective_created_at = created_at or datetime.now(tz=timezone.utc)
    effective_labeled_rows = list(labeled_rows)
    effective_unlabeled_rows = list(unlabeled_rows)
    if not effective_labeled_rows:
        raise ValueError("Query SSL LoRA local training requires labeled_rows.")
    if not effective_unlabeled_rows:
        raise ValueError("Query SSL LoRA local training requires unlabeled_rows.")

    descriptor = resolve_query_ssl_algorithm_descriptor(query_ssl_config.algorithm_name)
    validate_query_ssl_unlabeled_views(
        rows=effective_unlabeled_rows,
        view_builder_name=descriptor.required_views.view_builder_name,
        algorithm_name=descriptor.algorithm_name,
    )
    algorithm = descriptor.build_algorithm(query_ssl_config.parameters)
    labels = tuple(str(label) for label in active_adapter_state.label_schema)
    if not labels:
        raise ValueError("LoraClassifierState.label_schema must not be empty.")
    _validate_labeled_rows_have_known_labels(
        rows=effective_labeled_rows,
        labels=labels,
    )

    set_seed(int(seed))
    model, tokenizer = _build_lora_classifier_model(
        labels=labels,
        lora_config=lora_config,
        trainer_runtime_config=trainer_runtime_config,
    )
    base_parameters = _load_base_parameters(
        active_adapter_state=active_adapter_state,
        output_dir=output_dir,
        aggregated_at=effective_created_at,
    )
    load_lora_classifier_base_parameters_into_model(
        model=model,
        labels=labels,
        base_parameters=base_parameters,
        device=trainer_runtime_config.device,
    )

    label_to_index = {label: index for index, label in enumerate(labels)}
    train_loader = build_dataloader(
        rows=effective_labeled_rows,
        label_to_index=label_to_index,
        tokenizer=tokenizer,
        batch_size=int(training_task.batch_size),
        max_length=lora_config.max_length,
        task_prefix=lora_config.task_prefix,
        shuffle=True,
    )
    selection_loader = build_dataloader(
        rows=effective_labeled_rows,
        label_to_index=label_to_index,
        tokenizer=tokenizer,
        batch_size=int(training_task.batch_size),
        max_length=lora_config.max_length,
        task_prefix=lora_config.task_prefix,
        shuffle=False,
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
    )
    step_plan = build_query_ssl_local_step_plan(
        labeled_loader_steps=len(train_loader),
        unlabeled_loader_steps=len(unlabeled_loader),
        uses_labeled_batches=algorithm.uses_labeled_batches,
        local_epochs=int(training_task.local_epochs),
        max_steps=int(training_task.max_steps),
    )

    model, history, _best_selection_report = train_query_ssl_classifier(
        model=model,
        train_loader=train_loader,
        unlabeled_loader=unlabeled_loader,
        selection_loader=selection_loader,
        categories=list(labels),
        device=trainer_runtime_config.device,
        epochs=int(training_task.local_epochs),
        max_train_steps=step_plan.total_steps,
        learning_rate=float(training_task.learning_rate),
        classifier_learning_rate=float(training_task.learning_rate),
        weight_decay=0.0,
        max_grad_norm=(
            0.0
            if training_task.gradient_clip_norm is None
            else float(training_task.gradient_clip_norm)
        ),
        log_every_steps=0,
        algorithm=algorithm,
    )

    lora_deltas, head_weight_deltas, head_bias_deltas = (
        extract_lora_classifier_parameter_deltas(
            model=model,
            base_parameters=base_parameters,
            labels=labels,
        )
    )
    update_id = f"update_{training_task.round_id}_{client_id}_{uuid4().hex[:12]}"
    delta_materialization = _prepare_delta_materialization(
        output_dir=output_dir,
        update_id=update_id,
        training_task=training_task,
        client_id=client_id,
        delta_format=lora_config.delta_format,
        lora_parameter_deltas=lora_deltas,
        classifier_head_weight_deltas=head_weight_deltas,
        classifier_head_bias_deltas=head_bias_deltas,
    )
    update_build_result = build_query_ssl_lora_update_payload(
        training_task=training_task,
        model_manifest=model_manifest,
        lora_config=lora_config,
        labels=labels,
        labeled_rows=effective_labeled_rows,
        unlabeled_rows=effective_unlabeled_rows,
        step_plan=step_plan,
        history_record=history[-1] if history else {},
        lora_parameter_deltas=lora_deltas,
        classifier_head_weight_deltas=head_weight_deltas,
        classifier_head_bias_deltas=head_bias_deltas,
        created_at=effective_created_at,
        delta_format=delta_materialization.delta_format,
        lora_delta_artifact_ref=delta_materialization.lora_delta_artifact_ref,
        classifier_head_delta_artifact_ref=(
            delta_materialization.classifier_head_delta_artifact_ref
        ),
        include_inline_deltas=delta_materialization.include_inline_deltas,
    )
    update_payload = update_build_result.update_payload
    client_metrics = update_build_result.client_metrics
    update_envelope = make_training_update_envelope(
        update_id=update_id,
        round_id=training_task.round_id,
        task_id=training_task.task_id,
        model_id=model_manifest.model_id,
        base_model_revision=model_manifest.model_revision,
        training_scope=training_task.training_scope,
        payload_ref=f"client-submission::{update_id}",
        payload_format=LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
        example_count=update_payload.example_count,
        client_metrics=dict(client_metrics),
        created_at=effective_created_at,
    )
    return QuerySslLoraClientTrainingResult(
        update_envelope=update_envelope,
        update_payload=update_payload,
        candidate_count=len(effective_unlabeled_rows),
        accepted_count=update_build_result.accepted_unlabeled_count,
        local_step_plan=step_plan,
        client_metrics=client_metrics,
    )


def _build_lora_classifier_model(
    *,
    labels: Sequence[str],
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: FederatedLocalTrainerRuntimeConfig,
) -> tuple[LoraTextClassifier, Any]:
    AutoModel, AutoTokenizer, LoraConfig, TaskType, get_peft_model, _PeftModel = (
        require_transformer_stack()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        lora_config.tokenizer_model_id,
        revision=lora_config.tokenizer_revision,
        cache_dir=trainer_runtime_config.cache_dir,
        local_files_only=trainer_runtime_config.local_files_only,
        trust_remote_code=trainer_runtime_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    backbone_base = AutoModel.from_pretrained(
        lora_config.backbone_model_id,
        revision=lora_config.backbone_revision,
        cache_dir=trainer_runtime_config.cache_dir,
        local_files_only=trainer_runtime_config.local_files_only,
        trust_remote_code=trainer_runtime_config.trust_remote_code,
    )
    peft_config = LoraConfig(
        r=int(lora_config.rank),
        lora_alpha=int(lora_config.alpha),
        lora_dropout=float(lora_config.dropout),
        target_modules=resolve_target_modules(lora_config.target_modules),
        bias=lora_config.bias,
        use_rslora=bool(lora_config.use_rslora),
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    backbone = get_peft_model(backbone_base, peft_config)
    model = LoraTextClassifier(
        backbone=backbone,
        hidden_size=int(backbone.config.hidden_size),
        num_labels=len(labels),
        classifier_dropout=float(trainer_runtime_config.classifier_dropout),
    ).to(trainer_runtime_config.device)
    return model, tokenizer


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


def _prepare_delta_materialization(
    *,
    output_dir: Path,
    update_id: str,
    training_task: TrainingTask,
    client_id: str,
    delta_format: str,
    lora_parameter_deltas: Mapping[str, Sequence[float]],
    classifier_head_weight_deltas: Mapping[str, Sequence[float]],
    classifier_head_bias_deltas: Mapping[str, float],
) -> _DeltaMaterializationPlan:
    normalized_delta_format = str(delta_format).strip()
    if normalized_delta_format == LORA_CLASSIFIER_DELTA_FORMAT_INLINE:
        return _DeltaMaterializationPlan(
            delta_format=normalized_delta_format,
            lora_delta_artifact_ref=None,
            classifier_head_delta_artifact_ref=None,
            include_inline_deltas=True,
        )
    if normalized_delta_format != LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED:
        raise ValueError(
            f"Unsupported Query SSL LoRA delta_format: {normalized_delta_format!r}."
        )

    store = AggregationArtifactStore(
        state_root=output_dir / "main_server" / "aggregation_artifacts"
    )
    artifact_id_prefix = (
        f"client_updates/{training_task.round_id}/{client_id}/{update_id}"
    )
    lora_delta_ref = store.ref_for_artifact(f"{artifact_id_prefix}/lora_delta")
    head_delta_ref = store.ref_for_artifact(
        f"{artifact_id_prefix}/classifier_head_delta"
    )
    store.save_json_artifact_ref(
        artifact_ref=lora_delta_ref,
        payload={
            "schema_version": "lora_classifier_client_delta_artifact.v1",
            "update_id": update_id,
            "round_id": training_task.round_id,
            "client_id": client_id,
            "lora_parameter_deltas": {
                str(key): [float(value) for value in values]
                for key, values in lora_parameter_deltas.items()
            },
        },
    )
    store.save_json_artifact_ref(
        artifact_ref=head_delta_ref,
        payload={
            "schema_version": "lora_classifier_client_head_delta_artifact.v1",
            "update_id": update_id,
            "round_id": training_task.round_id,
            "client_id": client_id,
            "classifier_head_weight_deltas": {
                str(key): [float(value) for value in values]
                for key, values in classifier_head_weight_deltas.items()
            },
            "classifier_head_bias_deltas": {
                str(key): float(value)
                for key, value in classifier_head_bias_deltas.items()
            },
        },
    )
    return _DeltaMaterializationPlan(
        delta_format=normalized_delta_format,
        lora_delta_artifact_ref=lora_delta_ref,
        classifier_head_delta_artifact_ref=head_delta_ref,
        include_inline_deltas=False,
    )


def _build_unlabeled_loader(
    *,
    rows: Sequence[LabeledQueryRow],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    task_prefix: str,
    strong_view_policy: str,
    view_builder_name: str,
) -> Any:
    if view_builder_name == USB_MULTIVIEW_BUILDER_NAME:
        return build_multiview_dataloader(
            rows=rows,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            task_prefix=task_prefix,
            shuffle=True,
            strong_view_policy=strong_view_policy,
        )
    if view_builder_name == USB_WEAK_BUILDER_NAME:
        return build_weak_dataloader(
            rows=rows,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            task_prefix=task_prefix,
            shuffle=True,
        )
    raise ValueError(f"Unsupported Query SSL view builder: {view_builder_name}.")


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

"""Query SSL LoRA-classifier local training core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from methods.adaptation.lora.lora_adapter import resolve_target_modules
from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
)
from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.lora_classifier.update.query_ssl_update import (
    build_query_ssl_lora_update_payload,
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
from methods.ssl.registry import resolve_query_ssl_algorithm_descriptor
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    LoraClassifierDelta,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingTask,
    TrainingUpdateEnvelope,
    make_training_update_envelope,
)

from .delta_extraction import (
    extract_lora_classifier_parameter_deltas,
    load_lora_classifier_base_parameters_into_model,
)
from .loops import set_seed, train_query_ssl_classifier
from .modeling import LoraTextClassifier, require_transformer_stack


class QuerySslLoraObjectiveRuntimeConfig(Protocol):
    """Query SSL LoRA local core가 필요한 objective config surface."""

    algorithm_name: str
    parameters: Mapping[str, object]
    strong_view_policy: str
    unlabeled_batch_size: int | None


class LoraClassifierTrainerRuntimeConfig(Protocol):
    """LoRA classifier 모델 로딩/학습 core가 필요한 runtime config surface."""

    device: str
    classifier_dropout: float
    cache_dir: str | None
    local_files_only: bool
    trust_remote_code: bool


@dataclass(frozen=True, slots=True)
class QuerySslLoraDeltaMaterialization:
    """LoRA/head delta가 update payload에 담기는 방식."""

    delta_format: str
    lora_delta_artifact_ref: str | None
    classifier_head_delta_artifact_ref: str | None
    include_inline_deltas: bool


class QuerySslLoraDeltaMaterializer(Protocol):
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
    ) -> QuerySslLoraDeltaMaterialization:
        """delta 저장 방식을 결정하고 artifact ref를 반환한다."""


@dataclass(frozen=True, slots=True)
class QuerySslLoraClientTrainingResult:
    """FL round loop가 서버 제출과 client summary에 쓰는 local training 결과."""

    update_envelope: TrainingUpdateEnvelope
    update_payload: LoraClassifierDelta
    candidate_count: int
    accepted_count: int
    local_step_plan: QuerySslLocalStepPlan
    client_metrics: Mapping[str, float]


def run_query_ssl_lora_classifier_training_core(
    *,
    client_id: str,
    seed: int,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
    base_parameters: LoraClassifierMaterializedState,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    query_ssl_config: QuerySslLoraObjectiveRuntimeConfig,
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: LoraClassifierTrainerRuntimeConfig,
    created_at: datetime,
    delta_materializer: QuerySslLoraDeltaMaterializer,
) -> QuerySslLoraClientTrainingResult:
    """client-local raw text/views로 Query SSL LoRA update를 생성한다."""

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
    effective_labels = tuple(str(label) for label in labels)
    if not effective_labels:
        raise ValueError("LoRA classifier label schema must not be empty.")
    _validate_labeled_rows_have_known_labels(
        rows=effective_labeled_rows,
        labels=effective_labels,
    )

    set_seed(int(seed))
    model, tokenizer = _build_lora_classifier_model(
        labels=effective_labels,
        lora_config=lora_config,
        trainer_runtime_config=trainer_runtime_config,
    )
    load_lora_classifier_base_parameters_into_model(
        model=model,
        labels=effective_labels,
        base_parameters=base_parameters,
        device=trainer_runtime_config.device,
    )

    label_to_index = {label: index for index, label in enumerate(effective_labels)}
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
        categories=list(effective_labels),
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
            labels=effective_labels,
        )
    )
    update_id = f"update_{training_task.round_id}_{client_id}_{uuid4().hex[:12]}"
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
    update_build_result = build_query_ssl_lora_update_payload(
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
        created_at=created_at,
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
    trainer_runtime_config: LoraClassifierTrainerRuntimeConfig,
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

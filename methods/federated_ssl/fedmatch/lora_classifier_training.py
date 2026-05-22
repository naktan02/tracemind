"""FedMatch LoRA-classifier simulation bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

import torch

from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
)
from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.lora_classifier.training.delta_extraction import (
    extract_lora_classifier_parameter_deltas,
    load_lora_classifier_base_parameters_into_model,
)
from methods.adaptation.lora_classifier.training.loops import set_seed
from methods.adaptation.lora_classifier.training.modeling import (
    LoraTextClassifier,
    build_lora_text_classifier_from_config,
)
from methods.adaptation.lora_classifier.training.query_ssl_local_training import (
    LoraClassifierTrainerRuntimeConfig,
    QuerySslLoraClientTrainingResult,
    QuerySslLoraDeltaMaterializer,
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
    build_query_ssl_local_step_plan,
)
from methods.adaptation.query_classifier_adaptation.view_rows import (
    USB_MULTIVIEW_BUILDER_NAME,
    validate_query_ssl_unlabeled_views,
)
from methods.evaluation.pseudo_label_quality import (
    PseudoLabelCandidateRecord,
    PseudoLabelQualitySummary,
    build_pseudo_label_quality_summary,
)
from methods.federated_ssl.fedmatch.local_objective import (
    FedMatchLocalObjectiveParameters,
)
from methods.federated_ssl.fedmatch.lora_partitioned_loop import (
    train_fedmatch_lora_classifier,
)
from methods.federated_ssl.fedmatch.original_spec import (
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
)
from methods.federated_ssl.peer_context import FederatedSslPeerContext
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingTask,
    make_training_update_envelope,
)


class FedMatchMethodLocalTrainingConfig(Protocol):
    """FedMatch local trainer가 읽는 method config surface."""

    name: str
    scenario: str | None
    effective_parameters: Mapping[str, object]


def run_method_owned_lora_classifier_training_core(
    *,
    client_id: str,
    seed: int,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
    base_parameters: LoraClassifierMaterializedState,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    ssl_method_config: FedMatchMethodLocalTrainingConfig,
    strong_view_policy: str,
    unlabeled_batch_size: int | None,
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: LoraClassifierTrainerRuntimeConfig,
    created_at: datetime,
    delta_materializer: QuerySslLoraDeltaMaterializer,
    peer_context: FederatedSslPeerContext | None = None,
) -> QuerySslLoraClientTrainingResult:
    """FedMatch 원본 local objective를 LoRA-classifier simulation update로 실행한다."""

    scenario_name = ssl_method_config.scenario or FEDMATCH_SCENARIO_LABELS_AT_CLIENT
    if scenario_name != FEDMATCH_SCENARIO_LABELS_AT_CLIENT:
        raise NotImplementedError(
            "FedMatch LoRA local runtime slice currently supports only "
            f"{FEDMATCH_SCENARIO_LABELS_AT_CLIENT}; got {scenario_name!r}."
        )

    effective_labeled_rows = list(labeled_rows)
    effective_unlabeled_rows = list(unlabeled_rows)
    if not effective_labeled_rows:
        raise ValueError(
            "FedMatch labels-at-client local runtime requires labeled_rows."
        )
    if not effective_unlabeled_rows:
        raise ValueError("FedMatch local runtime requires unlabeled_rows.")

    validate_query_ssl_unlabeled_views(
        rows=effective_unlabeled_rows,
        view_builder_name=USB_MULTIVIEW_BUILDER_NAME,
        algorithm_name=str(ssl_method_config.name),
    )
    effective_labels = tuple(str(label) for label in labels)
    if not effective_labels:
        raise ValueError("FedMatch LoRA classifier label schema must not be empty.")
    _validate_labeled_rows_have_known_labels(
        rows=effective_labeled_rows,
        labels=effective_labels,
    )

    parameters = FedMatchLocalObjectiveParameters.from_mapping(
        ssl_method_config.effective_parameters
    )
    set_seed(int(seed))
    model, tokenizer = build_lora_text_classifier_from_config(
        labels=list(effective_labels),
        lora_config=lora_config,
        runtime_config=trainer_runtime_config,
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
    unlabeled_loader = build_multiview_dataloader(
        rows=effective_unlabeled_rows,
        tokenizer=tokenizer,
        batch_size=unlabeled_batch_size or int(training_task.batch_size),
        max_length=lora_config.max_length,
        task_prefix=lora_config.task_prefix,
        shuffle=True,
        strong_view_policy=strong_view_policy,
    )
    step_plan = build_query_ssl_local_step_plan(
        labeled_loader_steps=len(train_loader),
        unlabeled_loader_steps=len(unlabeled_loader),
        uses_labeled_batches=True,
        local_epochs=int(training_task.local_epochs),
        max_steps=int(training_task.max_steps),
    )

    training_result = train_fedmatch_lora_classifier(
        model=model,
        train_loader=train_loader,
        unlabeled_loader=unlabeled_loader,
        labels=effective_labels,
        parameters=parameters,
        step_plan=step_plan,
        device=trainer_runtime_config.device,
        learning_rate=float(training_task.learning_rate),
        classifier_learning_rate=float(training_task.learning_rate),
        weight_decay=0.0,
        max_grad_norm=(
            0.0
            if training_task.gradient_clip_norm is None
            else float(training_task.gradient_clip_norm)
        ),
        helper_weak_probability_provider=None,
        enable_inter_client_consistency=(
            peer_context is not None and peer_context.helper_count > 0
        ),
    )
    history_record = training_result.metrics
    pseudo_label_quality = _build_fedmatch_final_snapshot_pseudo_label_quality(
        model=model,
        tokenizer=tokenizer,
        rows=effective_unlabeled_rows,
        labels=effective_labels,
        lora_config=lora_config,
        parameters=parameters,
        trainer_runtime_config=trainer_runtime_config,
        unlabeled_batch_size=unlabeled_batch_size or int(training_task.batch_size),
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
        history_record=history_record,
        lora_parameter_deltas=lora_deltas,
        classifier_head_weight_deltas=head_weight_deltas,
        classifier_head_bias_deltas=head_bias_deltas,
        partitioned_deltas=training_result.partition_deltas,
        created_at=created_at,
        delta_format=delta_materialization.delta_format,
        lora_delta_artifact_ref=delta_materialization.lora_delta_artifact_ref,
        classifier_head_delta_artifact_ref=(
            delta_materialization.classifier_head_delta_artifact_ref
        ),
        include_inline_deltas=delta_materialization.include_inline_deltas,
    )
    peer_context_helper_count = (
        0.0 if peer_context is None else float(peer_context.helper_count)
    )
    client_metrics = {
        **dict(update_build_result.client_metrics),
        "fedmatch_local_runtime": 1.0,
        "fedmatch_helper_count": _history_float(
            history_record,
            "train_fedmatch_psi_helper_count",
        ),
        "fedmatch_peer_context_helper_count": peer_context_helper_count,
        "fedmatch_peer_context_refreshed": (
            0.0 if peer_context is None else float(peer_context.refreshed)
        ),
        "fedmatch_partitioned_delta_count": float(
            len(training_result.partition_deltas)
        ),
    }
    update_envelope = make_training_update_envelope(
        update_id=update_id,
        round_id=training_task.round_id,
        task_id=training_task.task_id,
        model_id=model_manifest.model_id,
        base_model_revision=model_manifest.model_revision,
        training_scope=training_task.training_scope,
        payload_ref=f"client-submission::{update_id}",
        payload_format=LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
        example_count=update_build_result.update_payload.example_count,
        client_metrics=client_metrics,
        created_at=created_at,
    )
    return QuerySslLoraClientTrainingResult(
        update_envelope=update_envelope,
        update_payload=update_build_result.update_payload,
        candidate_count=len(effective_unlabeled_rows),
        accepted_count=update_build_result.accepted_unlabeled_count,
        local_step_plan=step_plan,
        client_metrics=client_metrics,
        pseudo_label_quality=pseudo_label_quality,
    )


def _history_float(
    history_record: Mapping[str, object],
    key: str,
) -> float:
    value = history_record.get(key)
    if value is None:
        return 0.0
    return float(value)


def _build_fedmatch_final_snapshot_pseudo_label_quality(
    *,
    model: LoraTextClassifier,
    tokenizer: Any,
    rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
    lora_config: LoraClassifierTrainingBackendConfig,
    parameters: FedMatchLocalObjectiveParameters,
    trainer_runtime_config: LoraClassifierTrainerRuntimeConfig,
    unlabeled_batch_size: int,
) -> PseudoLabelQualitySummary:
    effective_rows = list(rows)
    if not effective_rows:
        return PseudoLabelQualitySummary.empty()

    loader = build_weak_dataloader(
        rows=effective_rows,
        tokenizer=tokenizer,
        batch_size=unlabeled_batch_size,
        max_length=lora_config.max_length,
        task_prefix=lora_config.task_prefix,
        shuffle=False,
    )
    candidates: list[PseudoLabelCandidateRecord] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["weak_input_ids"].to(trainer_runtime_config.device)
            attention_mask = batch["weak_attention_mask"].to(
                trainer_runtime_config.device
            )
            probabilities = torch.softmax(
                model(input_ids=input_ids, attention_mask=attention_mask),
                dim=-1,
            )
            top_k = min(2, probabilities.shape[-1])
            top_values, top_indices = torch.topk(probabilities, k=top_k, dim=-1)
            query_ids = [str(query_id) for query_id in batch["query_ids"]]
            for row_index, query_id in enumerate(query_ids):
                top1_index = int(top_indices[row_index, 0].detach().cpu().item())
                top1_score = float(top_values[row_index, 0].detach().cpu().item())
                top2_score = (
                    float(top_values[row_index, 1].detach().cpu().item())
                    if top_k > 1
                    else 0.0
                )
                candidates.append(
                    PseudoLabelCandidateRecord(
                        source_event_ref=query_id,
                        label=str(labels[top1_index]),
                        confidence=top1_score,
                        margin=top1_score - top2_score,
                        accepted=top1_score >= parameters.confidence_threshold,
                    )
                )

    return build_pseudo_label_quality_summary(
        candidates=tuple(candidates),
        rows_with_simulation_labels=effective_rows,
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
            "FedMatch labeled_rows contain labels outside active label_schema: "
            f"{missing}."
        )

"""FedMatch LoRA-classifier local training core."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from typing import Any, Protocol
from uuid import uuid4

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

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
from methods.adaptation.lora_classifier.training.loops import (
    build_optimizer,
    set_seed,
)
from methods.adaptation.lora_classifier.training.modeling import (
    LoraTextClassifier,
    build_lora_text_classifier_from_config,
)
from methods.adaptation.lora_classifier.training.partitioned_deltas import (
    build_lora_classifier_partition_delta_from_parameter_deltas,
    diff_parameter_snapshots,
    named_trainable_parameter_tensors,
    snapshot_trainable_parameter_tensors,
)
from methods.adaptation.lora_classifier.training.query_ssl_local_training import (
    LoraClassifierTrainerRuntimeConfig,
    QuerySslLoraClientTrainingResult,
    QuerySslLoraDeltaMaterializer,
)
from methods.adaptation.lora_classifier.update.partitioned_delta import (
    LoraClassifierPartitionDelta,
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
    validate_query_ssl_unlabeled_views,
)
from methods.evaluation.pseudo_label_quality import (
    PseudoLabelCandidateRecord,
    PseudoLabelQualitySummary,
    build_pseudo_label_quality_summary,
)
from methods.federated_ssl.fedmatch.local_objective import (
    FedMatchLocalObjectiveParameters,
    FedMatchParameterPartitions,
    FedMatchTensorLocalObjectiveResult,
    compute_fedmatch_supervised_loss,
    compute_fedmatch_unsupervised_loss,
)
from methods.federated_ssl.fedmatch.original_spec import (
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
)
from methods.federated_ssl.fedmatch.parameter_routing import (
    FEDMATCH_PSI_PARTITION,
    FEDMATCH_SIGMA_PARTITION,
)
from methods.federated_ssl.peer_context import FederatedSslPeerContext
from methods.ssl.base import TextBatchClassifier
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingTask,
    make_training_update_envelope,
)
from shared.src.domain.services.classification_report import safe_divide


@dataclass(frozen=True, slots=True)
class FedMatchLoraPartitionedStepResult:
    """FedMatch 한 step에서 분리 적용한 sigma/psi delta와 loss 진단."""

    supervised: FedMatchTensorLocalObjectiveResult
    unsupervised: FedMatchTensorLocalObjectiveResult
    sigma_parameter_deltas: Mapping[str, Tensor]
    psi_parameter_deltas: Mapping[str, Tensor]
    partition_deltas: Mapping[str, LoraClassifierPartitionDelta]
    metrics: Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class FedMatchLoraTrainingResult:
    """FedMatch local loop 결과와 누적 partition delta."""

    metrics: Mapping[str, float]
    partition_deltas: Mapping[str, LoraClassifierPartitionDelta]


class FedMatchMethodLocalTrainingConfig(Protocol):
    """FedMatch local trainer가 읽는 method config surface."""

    name: str
    scenario: str | None
    effective_parameters: Mapping[str, object]


class FedMatchHelperWeakProbabilityProvider(Protocol):
    """batch별 helper weak-view probability를 공급하는 runtime seam."""

    def __call__(
        self,
        *,
        unlabeled_batch: Mapping[str, Tensor],
    ) -> Tensor | Sequence[Tensor] | None:
        """helper model들이 현재 client batch에 낸 weak-view 확률을 반환한다."""


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


def train_fedmatch_lora_classifier(
    *,
    model: LoraTextClassifier,
    train_loader: DataLoader[dict[str, Any]],
    unlabeled_loader: DataLoader[dict[str, Any]],
    labels: Sequence[str],
    parameters: FedMatchLocalObjectiveParameters,
    step_plan: QuerySslLocalStepPlan,
    device: str,
    learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    helper_weak_probability_provider: (
        FedMatchHelperWeakProbabilityProvider | None
    ) = None,
    enable_inter_client_consistency: bool = True,
) -> FedMatchLoraTrainingResult:
    """FedMatch supervised/unsupervised 분리 step을 budget만큼 실행한다."""

    sigma_optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        weight_decay=weight_decay,
    )
    psi_optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = int(step_plan.total_steps)
    if total_steps <= 0:
        raise ValueError("FedMatch local step_plan.total_steps must be positive.")
    steps_per_epoch_budget = max(
        1,
        ceil(total_steps / max(1, int(step_plan.local_epochs))),
    )
    completed_steps = 0
    scalar_sums: dict[str, float] = {}
    sigma_parameter_delta_sums: dict[str, Tensor] = {}
    psi_parameter_delta_sums: dict[str, Tensor] = {}

    for _epoch in range(1, min(int(step_plan.local_epochs), total_steps) + 1):
        model.train()
        labeled_iterator = iter(train_loader)
        unlabeled_iterator = iter(unlabeled_loader)
        epoch_steps = min(steps_per_epoch_budget, total_steps - completed_steps)
        for _step_index in range(1, epoch_steps + 1):
            labeled_batch, labeled_iterator = _next_batch(
                loader=train_loader,
                iterator=labeled_iterator,
            )
            unlabeled_batch, unlabeled_iterator = _next_batch(
                loader=unlabeled_loader,
                iterator=unlabeled_iterator,
            )
            device_unlabeled_batch = _move_tensor_batch_to_device(
                batch=unlabeled_batch,
                device=device,
            )
            helper_weak_probabilities = (
                None
                if helper_weak_probability_provider is None
                else helper_weak_probability_provider(
                    unlabeled_batch=device_unlabeled_batch,
                )
            )
            step_result = run_fedmatch_lora_classifier_partitioned_step(
                model=model,
                labels=labels,
                labeled_batch=_move_tensor_batch_to_device(
                    batch=labeled_batch,
                    device=device,
                ),
                unlabeled_batch=device_unlabeled_batch,
                parameters=parameters,
                sigma_optimizer=sigma_optimizer,
                psi_optimizer=psi_optimizer,
                helper_weak_probabilities=helper_weak_probabilities,
                enable_inter_client_consistency=(
                    enable_inter_client_consistency
                    and helper_weak_probabilities is not None
                ),
                max_grad_norm=max_grad_norm,
            )
            completed_steps += 1
            _accumulate_named_scalar(
                scalar_sums,
                "sup_loss",
                step_result.supervised.total_loss,
            )
            _accumulate_named_scalar(
                scalar_sums,
                "unsup_loss",
                step_result.unsupervised.total_loss,
            )
            _accumulate_named_scalar(
                scalar_sums,
                "total_loss",
                step_result.supervised.total_loss + step_result.unsupervised.total_loss,
            )
            _accumulate_named_scalar(
                scalar_sums,
                "util_ratio",
                step_result.unsupervised.metrics["util_ratio"],
            )
            _accumulate_mapping_scalars(
                scalar_sums,
                prefix="fedmatch_sigma",
                values=step_result.supervised.metrics,
            )
            _accumulate_mapping_scalars(
                scalar_sums,
                prefix="fedmatch_psi",
                values=step_result.unsupervised.metrics,
            )
            _accumulate_named_float(
                scalar_sums,
                "fedmatch_sigma_delta_l2",
                _parameter_delta_l2(step_result.sigma_parameter_deltas),
            )
            _accumulate_named_float(
                scalar_sums,
                "fedmatch_psi_delta_l2",
                _parameter_delta_l2(step_result.psi_parameter_deltas),
            )
            _accumulate_parameter_deltas(
                sigma_parameter_delta_sums,
                step_result.sigma_parameter_deltas,
            )
            _accumulate_parameter_deltas(
                psi_parameter_delta_sums,
                step_result.psi_parameter_deltas,
            )
        if completed_steps >= total_steps:
            break

    return FedMatchLoraTrainingResult(
        metrics={
            f"train_{name}": round(safe_divide(value, completed_steps), 6)
            for name, value in scalar_sums.items()
        },
        partition_deltas={
            FEDMATCH_SIGMA_PARTITION: (
                build_lora_classifier_partition_delta_from_parameter_deltas(
                    partition_name=FEDMATCH_SIGMA_PARTITION,
                    parameter_deltas=sigma_parameter_delta_sums,
                    labels=labels,
                )
            ),
            FEDMATCH_PSI_PARTITION: (
                build_lora_classifier_partition_delta_from_parameter_deltas(
                    partition_name=FEDMATCH_PSI_PARTITION,
                    parameter_deltas=psi_parameter_delta_sums,
                    labels=labels,
                )
            ),
        },
    )


def run_fedmatch_lora_classifier_partitioned_step(
    *,
    model: TextBatchClassifier,
    labels: Sequence[str],
    labeled_batch: Mapping[str, Tensor],
    unlabeled_batch: Mapping[str, Tensor],
    parameters: FedMatchLocalObjectiveParameters,
    sigma_optimizer: torch.optim.Optimizer,
    psi_optimizer: torch.optim.Optimizer,
    helper_weak_probabilities: Tensor | Sequence[Tensor] | None = None,
    enable_inter_client_consistency: bool = True,
    max_grad_norm: float = 0.0,
) -> FedMatchLoraPartitionedStepResult:
    """원본 FedMatch처럼 supervised와 unsupervised update를 분리 적용한다.

    TraceMind의 LoRA-classifier 모델은 실제 parameter를 `sigma + psi`로 두 벌
    보관하지 않는다. 대신 같은 trainable tensor에 supervised step과 unsupervised
    step을 순차 적용하고, 두 sub-step의 delta를 logical `sigma`/`psi` partition으로
    분리해 기록한다.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("FedMatch LoRA partitioned step requires a torch nn.Module.")

    before_supervised = snapshot_trainable_parameter_tensors(model)
    supervised = _apply_fedmatch_supervised_step(
        model=model,
        labeled_batch=labeled_batch,
        parameters=parameters,
        optimizer=sigma_optimizer,
        max_grad_norm=max_grad_norm,
    )
    after_supervised = snapshot_trainable_parameter_tensors(model)
    sigma_parameter_deltas = diff_parameter_snapshots(
        after=after_supervised,
        before=before_supervised,
    )

    unsupervised = _apply_fedmatch_unsupervised_step(
        model=model,
        sigma_snapshot=after_supervised,
        unlabeled_batch=unlabeled_batch,
        parameters=parameters,
        optimizer=psi_optimizer,
        helper_weak_probabilities=helper_weak_probabilities,
        enable_inter_client_consistency=enable_inter_client_consistency,
        max_grad_norm=max_grad_norm,
    )
    after_unsupervised = snapshot_trainable_parameter_tensors(model)
    psi_parameter_deltas = diff_parameter_snapshots(
        after=after_unsupervised,
        before=after_supervised,
    )

    sigma_delta = build_lora_classifier_partition_delta_from_parameter_deltas(
        partition_name=FEDMATCH_SIGMA_PARTITION,
        parameter_deltas=sigma_parameter_deltas,
        labels=labels,
    )
    psi_delta = build_lora_classifier_partition_delta_from_parameter_deltas(
        partition_name=FEDMATCH_PSI_PARTITION,
        parameter_deltas=psi_parameter_deltas,
        labels=labels,
    )
    return FedMatchLoraPartitionedStepResult(
        supervised=supervised,
        unsupervised=unsupervised,
        sigma_parameter_deltas=sigma_parameter_deltas,
        psi_parameter_deltas=psi_parameter_deltas,
        partition_deltas={
            FEDMATCH_SIGMA_PARTITION: sigma_delta,
            FEDMATCH_PSI_PARTITION: psi_delta,
        },
        metrics={
            **{f"sigma_{key}": value for key, value in supervised.metrics.items()},
            **{f"psi_{key}": value for key, value in unsupervised.metrics.items()},
        },
    )


def _apply_fedmatch_supervised_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: Mapping[str, Tensor],
    parameters: FedMatchLocalObjectiveParameters,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
) -> FedMatchTensorLocalObjectiveResult:
    optimizer.zero_grad(set_to_none=True)
    logits = model(
        input_ids=labeled_batch["input_ids"],
        attention_mask=labeled_batch["attention_mask"],
    )
    result = compute_fedmatch_supervised_loss(
        labeled_logits=logits,
        labels=labeled_batch["labels"],
        parameters=parameters,
    )
    result.total_loss.backward()
    _clip_gradients_if_needed(
        model=model,
        max_grad_norm=max_grad_norm,
    )
    optimizer.step()
    return result


def _apply_fedmatch_unsupervised_step(
    *,
    model: TextBatchClassifier,
    sigma_snapshot: Mapping[str, Tensor],
    unlabeled_batch: Mapping[str, Tensor],
    parameters: FedMatchLocalObjectiveParameters,
    optimizer: torch.optim.Optimizer,
    helper_weak_probabilities: Tensor | Sequence[Tensor] | None,
    enable_inter_client_consistency: bool,
    max_grad_norm: float,
) -> FedMatchTensorLocalObjectiveResult:
    optimizer.zero_grad(set_to_none=True)
    weak_logits = model(
        input_ids=unlabeled_batch["weak_input_ids"],
        attention_mask=unlabeled_batch["weak_attention_mask"],
    )
    strong_logits = model(
        input_ids=unlabeled_batch["strong_input_ids"],
        attention_mask=unlabeled_batch["strong_attention_mask"],
    )
    result = compute_fedmatch_unsupervised_loss(
        weak_logits=weak_logits,
        strong_logits=strong_logits,
        helper_weak_probabilities=helper_weak_probabilities,
        parameter_partitions=FedMatchParameterPartitions(
            sigma={
                key: value.to(device=weak_logits.device, dtype=weak_logits.dtype)
                for key, value in sigma_snapshot.items()
            },
            psi=named_trainable_parameter_tensors(_as_torch_module(model)),
        ),
        parameters=parameters,
        enable_inter_client_consistency=enable_inter_client_consistency,
    )
    result.total_loss.backward()
    _clip_gradients_if_needed(
        model=model,
        max_grad_norm=max_grad_norm,
    )
    optimizer.step()
    return result


def _clip_gradients_if_needed(
    *,
    model: TextBatchClassifier,
    max_grad_norm: float,
) -> None:
    if max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(
            _as_torch_module(model).parameters(),
            max_grad_norm,
        )


def _as_torch_module(model: TextBatchClassifier) -> nn.Module:
    return model  # type: ignore[return-value]


def _next_batch(
    *,
    loader: DataLoader[dict[str, Any]],
    iterator: Iterator[dict[str, Any]],
) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        refreshed_iterator = iter(loader)
        return next(refreshed_iterator), refreshed_iterator


def _move_tensor_batch_to_device(
    *,
    batch: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def _accumulate_named_scalar(
    sums: dict[str, float],
    name: str,
    value: Tensor,
) -> None:
    _accumulate_named_float(sums, name, float(value.detach().item()))


def _accumulate_named_float(
    sums: dict[str, float],
    name: str,
    value: float,
) -> None:
    sums[name] = sums.get(name, 0.0) + float(value)


def _accumulate_mapping_scalars(
    sums: dict[str, float],
    *,
    prefix: str,
    values: Mapping[str, Tensor],
) -> None:
    for name, value in values.items():
        _accumulate_named_scalar(sums, f"{prefix}_{name}", value)


def _accumulate_parameter_deltas(
    sums: dict[str, Tensor],
    values: Mapping[str, Tensor],
) -> None:
    for name, value in values.items():
        detached = value.detach().cpu()
        if name not in sums:
            sums[name] = detached.clone()
            continue
        if sums[name].shape != detached.shape:
            raise ValueError("FedMatch partition delta shape changed during training.")
        sums[name] = sums[name] + detached


def _parameter_delta_l2(parameter_deltas: Mapping[str, Tensor]) -> float:
    squared_norm = 0.0
    for delta in parameter_deltas.values():
        squared_norm += float(torch.sum(torch.square(delta.detach())).item())
    return squared_norm**0.5


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

"""LoRA-classifier family의 partitioned local training loop."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from methods.adaptation.lora_classifier.training.loops import build_optimizer
from methods.adaptation.lora_classifier.training.modeling import LoraTextClassifier
from methods.adaptation.lora_classifier.training.partitioned_deltas import (
    build_lora_classifier_partition_delta_from_parameter_deltas,
    diff_parameter_snapshots,
    named_trainable_parameter_tensors,
    snapshot_trainable_parameter_tensors,
)
from methods.adaptation.lora_classifier.update.partitioned_delta import (
    LoraClassifierPartitionDelta,
)
from methods.adaptation.query_classifier_adaptation.local_training_budget import (
    QuerySslLocalStepPlan,
)
from methods.federated_ssl.fedmatch.local_objective import (
    FedMatchLocalObjectiveParameters,
    FedMatchParameterPartitions,
    FedMatchTensorLocalObjectiveResult,
    compute_fedmatch_supervised_loss,
    compute_fedmatch_unsupervised_loss,
)
from methods.federated_ssl.fedmatch.parameter_routing import (
    FEDMATCH_PSI_PARTITION,
    FEDMATCH_SIGMA_PARTITION,
)
from methods.ssl.base import QuerySslAlgorithm, QuerySslStepResult, TextBatchClassifier
from shared.src.domain.services.classification_report import safe_divide


@dataclass(frozen=True, slots=True)
class PartitionedLoraStepResult:
    """한 step에서 분리 적용한 partition delta와 loss 진단."""

    supervised: FedMatchTensorLocalObjectiveResult
    unsupervised: FedMatchTensorLocalObjectiveResult
    sigma_parameter_deltas: Mapping[str, Tensor]
    psi_parameter_deltas: Mapping[str, Tensor]
    metrics: Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class PartitionedLoraTrainingResult:
    """partitioned local loop 결과와 누적 partition delta."""

    metrics: Mapping[str, float]
    partition_deltas: Mapping[str, LoraClassifierPartitionDelta]


class HelperWeakProbabilityProvider(Protocol):
    """batch별 helper weak-view probability를 공급하는 runtime seam."""

    def __call__(
        self,
        *,
        unlabeled_batch: Mapping[str, Tensor],
    ) -> Tensor | Sequence[Tensor] | None:
        """helper model들이 현재 client batch에 낸 weak-view 확률을 반환한다."""


def train_partitioned_lora_classifier(
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
    helper_weak_probability_provider: (HelperWeakProbabilityProvider | None) = None,
    psi_query_ssl_algorithm: QuerySslAlgorithm | None = None,
    enable_inter_client_consistency: bool = True,
) -> PartitionedLoraTrainingResult:
    """supervised/unsupervised partitioned step을 budget만큼 실행한다."""

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
            step_result = run_partitioned_lora_classifier_step(
                model=model,
                labeled_batch=_move_tensor_batch_to_device(
                    batch=labeled_batch,
                    device=device,
                ),
                unlabeled_batch=device_unlabeled_batch,
                parameters=parameters,
                sigma_optimizer=sigma_optimizer,
                psi_optimizer=psi_optimizer,
                helper_weak_probabilities=helper_weak_probabilities,
                psi_query_ssl_algorithm=psi_query_ssl_algorithm,
                enable_inter_client_consistency=(
                    enable_inter_client_consistency
                    and helper_weak_probabilities is not None
                    and psi_query_ssl_algorithm is None
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

    return PartitionedLoraTrainingResult(
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


def run_partitioned_lora_classifier_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: Mapping[str, Tensor],
    unlabeled_batch: Mapping[str, Tensor],
    parameters: FedMatchLocalObjectiveParameters,
    sigma_optimizer: torch.optim.Optimizer,
    psi_optimizer: torch.optim.Optimizer,
    helper_weak_probabilities: Tensor | Sequence[Tensor] | None = None,
    psi_query_ssl_algorithm: QuerySslAlgorithm | None = None,
    enable_inter_client_consistency: bool = True,
    max_grad_norm: float = 0.0,
) -> PartitionedLoraStepResult:
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

    unsupervised = (
        _apply_query_ssl_psi_step(
            model=model,
            unlabeled_batch=unlabeled_batch,
            optimizer=psi_optimizer,
            algorithm=psi_query_ssl_algorithm,
            max_grad_norm=max_grad_norm,
        )
        if psi_query_ssl_algorithm is not None
        else _apply_fedmatch_unsupervised_step(
            model=model,
            sigma_snapshot=after_supervised,
            unlabeled_batch=unlabeled_batch,
            parameters=parameters,
            optimizer=psi_optimizer,
            helper_weak_probabilities=helper_weak_probabilities,
            enable_inter_client_consistency=enable_inter_client_consistency,
            max_grad_norm=max_grad_norm,
        )
    )
    after_unsupervised = snapshot_trainable_parameter_tensors(model)
    psi_parameter_deltas = diff_parameter_snapshots(
        after=after_unsupervised,
        before=after_supervised,
    )

    return PartitionedLoraStepResult(
        supervised=supervised,
        unsupervised=unsupervised,
        sigma_parameter_deltas=sigma_parameter_deltas,
        psi_parameter_deltas=psi_parameter_deltas,
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


def _apply_query_ssl_psi_step(
    *,
    model: TextBatchClassifier,
    unlabeled_batch: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    algorithm: QuerySslAlgorithm,
    max_grad_norm: float,
) -> FedMatchTensorLocalObjectiveResult:
    optimizer.zero_grad(set_to_none=True)
    step_result = algorithm.compute_step(
        model=model,
        labeled_batch=None,
        unlabeled_batch=dict(unlabeled_batch),
    )
    step_result.total_loss.backward()
    _clip_gradients_if_needed(
        model=model,
        max_grad_norm=max_grad_norm,
    )
    optimizer.step()
    return _query_ssl_step_result_as_partition_result(step_result)


def _query_ssl_step_result_as_partition_result(
    step_result: QuerySslStepResult,
) -> FedMatchTensorLocalObjectiveResult:
    return FedMatchTensorLocalObjectiveResult(
        total_loss=step_result.total_loss,
        partition_losses={FEDMATCH_PSI_PARTITION: step_result.total_loss},
        loss_components=step_result.loss_components,
        metrics=step_result.metrics,
        debug_tensors=step_result.debug_tensors,
    )


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

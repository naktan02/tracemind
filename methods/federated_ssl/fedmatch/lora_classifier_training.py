"""FedMatch LoRA-classifier partitioned optimizer step core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from methods.adaptation.lora_classifier.training.partitioned_deltas import (
    build_lora_classifier_partition_delta_from_parameter_deltas,
    diff_parameter_snapshots,
    named_trainable_parameter_tensors,
    snapshot_trainable_parameter_tensors,
)
from methods.adaptation.lora_classifier.update.partitioned_delta import (
    LoraClassifierPartitionDelta,
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
from methods.ssl.base import TextBatchClassifier


@dataclass(frozen=True, slots=True)
class FedMatchLoraPartitionedStepResult:
    """FedMatch 한 step에서 분리 적용한 sigma/psi delta와 loss 진단."""

    supervised: FedMatchTensorLocalObjectiveResult
    unsupervised: FedMatchTensorLocalObjectiveResult
    sigma_parameter_deltas: Mapping[str, Tensor]
    psi_parameter_deltas: Mapping[str, Tensor]
    partition_deltas: Mapping[str, LoraClassifierPartitionDelta]
    metrics: Mapping[str, Tensor]


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

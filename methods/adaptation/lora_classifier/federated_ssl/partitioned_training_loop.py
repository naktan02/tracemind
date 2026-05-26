"""LoRA-classifier family의 partitioned local training loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from methods.adaptation.lora_classifier.federated_ssl import (
    partitioned_trainable_model as ptm,
)
from methods.adaptation.lora_classifier.training.batching import (
    move_tensor_batch_to_device,
    next_cycling_batch,
)
from methods.adaptation.lora_classifier.training.loops import (
    build_optimizer,
    trainable_model_parameters,
)
from methods.adaptation.lora_classifier.training.modeling import LoraTextClassifier
from methods.adaptation.lora_classifier.training.optimizer_step import (
    run_optimizer_loss_step,
)
from methods.adaptation.lora_classifier.training.partitioned_deltas import (
    build_lora_classifier_partition_delta_from_parameter_deltas,
    diff_parameter_snapshots,
    named_trainable_parameter_tensors,
    snapshot_trainable_parameter_tensors,
)
from methods.adaptation.lora_classifier.training.scalar_metrics import (
    ScalarMetricAccumulator,
    tensor_mapping_l2,
)
from methods.adaptation.lora_classifier.training.step_budget import (
    resolve_epoch_distributed_step_budget,
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
        """helper weak-view 확률을 반환한다."""


def train_partitioned_lora_classifier(
    *,
    model: LoraTextClassifier,
    train_loader: DataLoader[dict[str, Any]] | None,
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
    use_supervised_steps: bool = True,
    emit_sigma_partition: bool = True,
) -> PartitionedLoraTrainingResult:
    """supervised/unsupervised partitioned step을 budget만큼 실행한다."""

    if use_supervised_steps and train_loader is None:
        raise ValueError("supervised FedMatch steps require train_loader.")
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
    step_budget = resolve_epoch_distributed_step_budget(
        epochs=int(step_plan.local_epochs),
        full_epoch_steps=int(step_plan.full_epoch_steps),
        max_train_steps=total_steps,
        invalid_max_steps_message=(
            "FedMatch local step_plan.total_steps must be positive."
        ),
    )
    completed_steps = 0
    scalar_metrics = ScalarMetricAccumulator()
    sigma_parameter_delta_sums: dict[str, Tensor] = {}
    psi_parameter_delta_sums: dict[str, Tensor] = {}

    for _epoch in range(1, step_budget.effective_epochs + 1):
        model.train()
        labeled_iterator = iter(train_loader) if use_supervised_steps else None
        unlabeled_iterator = iter(unlabeled_loader)
        epoch_steps = step_budget.remaining_epoch_steps(completed_steps)
        for _step_index in range(1, epoch_steps + 1):
            labeled_batch = None
            if use_supervised_steps:
                if train_loader is None or labeled_iterator is None:
                    raise ValueError("supervised FedMatch steps require train_loader.")
                labeled_batch, labeled_iterator = next_cycling_batch(
                    loader=train_loader,
                    iterator=labeled_iterator,
                )
            unlabeled_batch, unlabeled_iterator = next_cycling_batch(
                loader=unlabeled_loader,
                iterator=unlabeled_iterator,
            )
            device_unlabeled_batch = move_tensor_batch_to_device(
                batch=unlabeled_batch,
                device=device,
            )
            step_result = run_partitioned_lora_classifier_step(
                model=model,
                labeled_batch=(
                    None
                    if labeled_batch is None
                    else move_tensor_batch_to_device(
                        batch=labeled_batch,
                        device=device,
                    )
                ),
                unlabeled_batch=device_unlabeled_batch,
                parameters=parameters,
                sigma_optimizer=sigma_optimizer,
                psi_optimizer=psi_optimizer,
                helper_weak_probability_provider=helper_weak_probability_provider,
                psi_query_ssl_algorithm=psi_query_ssl_algorithm,
                enable_inter_client_consistency=(
                    enable_inter_client_consistency
                    and helper_weak_probability_provider is not None
                    and psi_query_ssl_algorithm is None
                ),
                apply_supervised_step=use_supervised_steps,
                max_grad_norm=max_grad_norm,
            )
            completed_steps += 1
            scalar_metrics.add_tensor(
                "sup_loss",
                step_result.supervised.total_loss,
            )
            scalar_metrics.add_tensor(
                "unsup_loss",
                step_result.unsupervised.total_loss,
            )
            scalar_metrics.add_tensor(
                "total_loss",
                step_result.supervised.total_loss + step_result.unsupervised.total_loss,
            )
            scalar_metrics.add_tensor(
                "util_ratio",
                step_result.unsupervised.metrics["util_ratio"],
            )
            scalar_metrics.add_tensor_mapping(
                step_result.supervised.metrics,
                prefix="fedmatch_sigma_",
            )
            scalar_metrics.add_tensor_mapping(
                step_result.unsupervised.metrics,
                prefix="fedmatch_psi_",
            )
            scalar_metrics.add_float(
                "fedmatch_sigma_delta_l2",
                tensor_mapping_l2(step_result.sigma_parameter_deltas),
            )
            scalar_metrics.add_float(
                "fedmatch_psi_delta_l2",
                tensor_mapping_l2(step_result.psi_parameter_deltas),
            )
            if emit_sigma_partition:
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
        metrics=scalar_metrics.average_record(
            denominator=completed_steps,
            key_prefix="train_",
        ),
        partition_deltas={
            **(
                {
                    FEDMATCH_SIGMA_PARTITION: (
                        build_lora_classifier_partition_delta_from_parameter_deltas(
                            partition_name=FEDMATCH_SIGMA_PARTITION,
                            parameter_deltas=sigma_parameter_delta_sums,
                            labels=labels,
                        )
                    )
                }
                if emit_sigma_partition
                else {}
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


def train_physical_partitioned_adapter_classifier(
    *,
    model: ptm.PartitionedTrainableTextClassifier,
    train_loader: DataLoader[dict[str, Any]] | None,
    unlabeled_loader: DataLoader[dict[str, Any]],
    labels: Sequence[str],
    parameters: FedMatchLocalObjectiveParameters,
    step_plan: QuerySslLocalStepPlan,
    device: str,
    learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    supervised_partition: str,
    unsupervised_partition: str,
    helper_weak_probability_provider: (HelperWeakProbabilityProvider | None) = None,
    enable_inter_client_consistency: bool = True,
    use_supervised_steps: bool = True,
    emit_supervised_partition: bool = True,
) -> PartitionedLoraTrainingResult:
    """physical trainable partition loop를 budget만큼 실행한다.

    이 함수는 FedMatch 이름이나 concrete PEFT adapter 종류를 해석하지 않는다.
    caller가 지정한 supervised/unsupervised partition에 objective를 라우팅하고,
    현재 `lora_classifier` shared payload로 projection 가능한 partition delta를
    반환한다.
    """

    if use_supervised_steps and train_loader is None:
        raise ValueError("physical partition supervised steps require train_loader.")
    sigma_optimizer = _build_partition_optimizer(
        model=model,
        partition_name=supervised_partition,
        learning_rate=learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        weight_decay=weight_decay,
    )
    psi_optimizer = _build_partition_optimizer(
        model=model,
        partition_name=unsupervised_partition,
        learning_rate=learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = int(step_plan.total_steps)
    if total_steps <= 0:
        raise ValueError("physical partition step_plan.total_steps must be positive.")
    step_budget = resolve_epoch_distributed_step_budget(
        epochs=int(step_plan.local_epochs),
        full_epoch_steps=int(step_plan.full_epoch_steps),
        max_train_steps=total_steps,
        invalid_max_steps_message=(
            "physical partition step_plan.total_steps must be positive."
        ),
    )
    completed_steps = 0
    scalar_metrics = ScalarMetricAccumulator()
    supervised_parameter_delta_sums: dict[str, Tensor] = {}
    unsupervised_parameter_delta_sums: dict[str, Tensor] = {}

    for _epoch in range(1, step_budget.effective_epochs + 1):
        _train_if_module(model)
        labeled_iterator = iter(train_loader) if use_supervised_steps else None
        unlabeled_iterator = iter(unlabeled_loader)
        epoch_steps = step_budget.remaining_epoch_steps(completed_steps)
        for _step_index in range(1, epoch_steps + 1):
            labeled_batch = None
            if use_supervised_steps:
                if train_loader is None or labeled_iterator is None:
                    raise ValueError(
                        "physical partition supervised steps require train_loader."
                    )
                labeled_batch, labeled_iterator = next_cycling_batch(
                    loader=train_loader,
                    iterator=labeled_iterator,
                )
            unlabeled_batch, unlabeled_iterator = next_cycling_batch(
                loader=unlabeled_loader,
                iterator=unlabeled_iterator,
            )
            step_result = run_physical_partitioned_adapter_classifier_step(
                model=model,
                labeled_batch=(
                    None
                    if labeled_batch is None
                    else move_tensor_batch_to_device(
                        batch=labeled_batch,
                        device=device,
                    )
                ),
                unlabeled_batch=move_tensor_batch_to_device(
                    batch=unlabeled_batch,
                    device=device,
                ),
                parameters=parameters,
                supervised_partition=supervised_partition,
                unsupervised_partition=unsupervised_partition,
                sigma_optimizer=sigma_optimizer,
                psi_optimizer=psi_optimizer,
                helper_weak_probability_provider=helper_weak_probability_provider,
                enable_inter_client_consistency=(
                    enable_inter_client_consistency
                    and helper_weak_probability_provider is not None
                ),
                apply_supervised_step=use_supervised_steps,
                max_grad_norm=max_grad_norm,
            )
            completed_steps += 1
            scalar_metrics.add_tensor("sup_loss", step_result.supervised.total_loss)
            scalar_metrics.add_tensor("unsup_loss", step_result.unsupervised.total_loss)
            scalar_metrics.add_tensor(
                "total_loss",
                step_result.supervised.total_loss + step_result.unsupervised.total_loss,
            )
            scalar_metrics.add_tensor(
                "util_ratio",
                step_result.unsupervised.metrics["util_ratio"],
            )
            scalar_metrics.add_tensor_mapping(
                step_result.supervised.metrics,
                prefix="fedmatch_sigma_",
            )
            scalar_metrics.add_tensor_mapping(
                step_result.unsupervised.metrics,
                prefix="fedmatch_psi_",
            )
            scalar_metrics.add_float(
                "fedmatch_sigma_delta_l2",
                tensor_mapping_l2(step_result.sigma_parameter_deltas),
            )
            scalar_metrics.add_float(
                "fedmatch_psi_delta_l2",
                tensor_mapping_l2(step_result.psi_parameter_deltas),
            )
            if emit_supervised_partition:
                _accumulate_parameter_deltas(
                    supervised_parameter_delta_sums,
                    step_result.sigma_parameter_deltas,
                )
            _accumulate_parameter_deltas(
                unsupervised_parameter_delta_sums,
                step_result.psi_parameter_deltas,
            )
        if completed_steps >= total_steps:
            break

    return PartitionedLoraTrainingResult(
        metrics=scalar_metrics.average_record(
            denominator=completed_steps,
            key_prefix="train_",
        ),
        partition_deltas={
            **(
                {
                    supervised_partition: (
                        build_lora_classifier_partition_delta_from_parameter_deltas(
                            partition_name=supervised_partition,
                            parameter_deltas=supervised_parameter_delta_sums,
                            labels=labels,
                        )
                    )
                }
                if emit_supervised_partition
                else {}
            ),
            unsupervised_partition: (
                build_lora_classifier_partition_delta_from_parameter_deltas(
                    partition_name=unsupervised_partition,
                    parameter_deltas=unsupervised_parameter_delta_sums,
                    labels=labels,
                )
            ),
        },
    )


def run_partitioned_lora_classifier_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: Mapping[str, Tensor] | None,
    unlabeled_batch: Mapping[str, Tensor],
    parameters: FedMatchLocalObjectiveParameters,
    sigma_optimizer: torch.optim.Optimizer,
    psi_optimizer: torch.optim.Optimizer,
    helper_weak_probability_provider: HelperWeakProbabilityProvider | None = None,
    psi_query_ssl_algorithm: QuerySslAlgorithm | None = None,
    enable_inter_client_consistency: bool = True,
    apply_supervised_step: bool = True,
    max_grad_norm: float = 0.0,
) -> PartitionedLoraStepResult:
    """원본 FedMatch처럼 supervised와 unsupervised update를 분리 적용한다.

    TraceMind의 LoRA-classifier 모델은 실제 parameter를 `sigma + psi`로 두 벌
    보관하지 않는다. 같은 trainable tensor에 두 step을 순차 적용하고,
    sub-step delta를 `sigma`/`psi` partition으로 기록한다.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("FedMatch LoRA partitioned step requires a torch nn.Module.")
    if apply_supervised_step and labeled_batch is None:
        raise ValueError("FedMatch supervised step requires labeled_batch.")

    before_supervised = snapshot_trainable_parameter_tensors(model)
    if apply_supervised_step:
        if labeled_batch is None:
            raise ValueError("FedMatch supervised step requires labeled_batch.")
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
    else:
        supervised = _empty_supervised_step_result(before_supervised)
        after_supervised = before_supervised
        sigma_parameter_deltas = {}

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
            helper_weak_probability_provider=helper_weak_probability_provider,
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


def run_physical_partitioned_adapter_classifier_step(
    *,
    model: ptm.PartitionedTrainableTextClassifier,
    labeled_batch: Mapping[str, Tensor] | None,
    unlabeled_batch: Mapping[str, Tensor],
    parameters: FedMatchLocalObjectiveParameters,
    supervised_partition: str,
    unsupervised_partition: str,
    sigma_optimizer: torch.optim.Optimizer,
    psi_optimizer: torch.optim.Optimizer,
    helper_weak_probability_provider: HelperWeakProbabilityProvider | None = None,
    enable_inter_client_consistency: bool = True,
    apply_supervised_step: bool = True,
    max_grad_norm: float = 0.0,
) -> PartitionedLoraStepResult:
    """원본 의미의 sigma/psi를 물리적으로 분리한 partition에 적용한다.

    frozen backbone은 공유하고 adapter/head trainable state만 partition별로
    보관한다. 이 함수는 partition 이름의 FedMatch 의미를 해석하지 않고,
    caller가 지정한 supervised/unsupervised partition에 objective를 라우팅한다.
    """

    if apply_supervised_step and labeled_batch is None:
        raise ValueError("FedMatch physical supervised step requires labeled_batch.")

    before_sigma = ptm.snapshot_partition_parameter_tensors(
        model,
        supervised_partition,
    )
    if apply_supervised_step:
        if labeled_batch is None:
            raise ValueError(
                "FedMatch physical supervised step requires labeled_batch."
            )
        supervised = _apply_physical_fedmatch_supervised_step(
            model=model,
            partition_name=supervised_partition,
            labeled_batch=labeled_batch,
            parameters=parameters,
            optimizer=sigma_optimizer,
            max_grad_norm=max_grad_norm,
        )
        after_sigma = ptm.snapshot_partition_parameter_tensors(
            model,
            supervised_partition,
        )
        sigma_parameter_deltas = diff_parameter_snapshots(
            after=after_sigma,
            before=before_sigma,
        )
    else:
        supervised = _empty_supervised_step_result(before_sigma)
        after_sigma = before_sigma
        sigma_parameter_deltas = {}

    before_psi = ptm.snapshot_partition_parameter_tensors(
        model,
        unsupervised_partition,
    )
    unsupervised = _apply_physical_fedmatch_unsupervised_step(
        model=model,
        sigma_partition_snapshot=after_sigma,
        unsupervised_partition=unsupervised_partition,
        unlabeled_batch=unlabeled_batch,
        parameters=parameters,
        optimizer=psi_optimizer,
        helper_weak_probability_provider=helper_weak_probability_provider,
        enable_inter_client_consistency=enable_inter_client_consistency,
        max_grad_norm=max_grad_norm,
    )
    after_psi = ptm.snapshot_partition_parameter_tensors(
        model,
        unsupervised_partition,
    )
    psi_parameter_deltas = diff_parameter_snapshots(
        after=after_psi,
        before=before_psi,
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
    result: FedMatchTensorLocalObjectiveResult | None = None

    def compute_loss() -> Tensor:
        nonlocal result

        logits = model(
            input_ids=labeled_batch["input_ids"],
            attention_mask=labeled_batch["attention_mask"],
        )
        result = compute_fedmatch_supervised_loss(
            labeled_logits=logits,
            labels=labeled_batch["labels"],
            parameters=parameters,
        )
        return result.total_loss

    run_optimizer_loss_step(
        optimizer=optimizer,
        trainable_parameters=trainable_model_parameters(_as_torch_module(model)),
        max_grad_norm=max_grad_norm,
        compute_loss=compute_loss,
    )
    assert result is not None
    return result


def _apply_physical_fedmatch_supervised_step(
    *,
    model: ptm.PartitionedTrainableTextClassifier,
    partition_name: str,
    labeled_batch: Mapping[str, Tensor],
    parameters: FedMatchLocalObjectiveParameters,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
) -> FedMatchTensorLocalObjectiveResult:
    result: FedMatchTensorLocalObjectiveResult | None = None

    def compute_loss() -> Tensor:
        nonlocal result

        logits = model.forward_partition(
            partition_name,
            input_ids=labeled_batch["input_ids"],
            attention_mask=labeled_batch["attention_mask"],
        )
        result = compute_fedmatch_supervised_loss(
            labeled_logits=logits,
            labels=labeled_batch["labels"],
            parameters=parameters,
        )
        return result.total_loss

    run_optimizer_loss_step(
        optimizer=optimizer,
        trainable_parameters=model.partition_parameters(partition_name),
        max_grad_norm=max_grad_norm,
        compute_loss=compute_loss,
    )
    assert result is not None
    return result


def _empty_supervised_step_result(
    parameter_snapshot: Mapping[str, Tensor],
) -> FedMatchTensorLocalObjectiveResult:
    try:
        reference = next(iter(parameter_snapshot.values()))
    except StopIteration as error:
        raise ValueError(
            "FedMatch partitioned step requires trainable parameters."
        ) from error
    zero = reference.new_zeros(())
    return FedMatchTensorLocalObjectiveResult(
        total_loss=zero,
        partition_losses={FEDMATCH_SIGMA_PARTITION: zero},
        loss_components={},
        metrics={"labeled_count": reference.new_tensor(0.0)},
        debug_tensors={},
    )


def _apply_fedmatch_unsupervised_step(
    *,
    model: TextBatchClassifier,
    sigma_snapshot: Mapping[str, Tensor],
    unlabeled_batch: Mapping[str, Tensor],
    parameters: FedMatchLocalObjectiveParameters,
    optimizer: torch.optim.Optimizer,
    helper_weak_probability_provider: HelperWeakProbabilityProvider | None,
    enable_inter_client_consistency: bool,
    max_grad_norm: float,
) -> FedMatchTensorLocalObjectiveResult:
    result: FedMatchTensorLocalObjectiveResult | None = None

    def compute_loss() -> Tensor:
        nonlocal result

        weak_logits = model(
            input_ids=unlabeled_batch["weak_input_ids"],
            attention_mask=unlabeled_batch["weak_attention_mask"],
        )
        confidence_mask = _fedmatch_confidence_mask(
            weak_logits=weak_logits,
            confidence_threshold=parameters.confidence_threshold,
        )
        selected_strong_logits = _forward_selected_strong_view(
            model=model,
            unlabeled_batch=unlabeled_batch,
            confidence_mask=confidence_mask,
            reference_logits=weak_logits,
        )
        selected_helper_probabilities = _compute_selected_helper_probabilities(
            helper_weak_probability_provider=helper_weak_probability_provider,
            unlabeled_batch=unlabeled_batch,
            confidence_mask=confidence_mask,
        )
        result = compute_fedmatch_unsupervised_loss(
            weak_logits=weak_logits,
            selected_strong_logits=selected_strong_logits,
            selected_helper_weak_probabilities=selected_helper_probabilities,
            parameter_partitions=FedMatchParameterPartitions(
                sigma={
                    key: value.to(device=weak_logits.device, dtype=weak_logits.dtype)
                    for key, value in sigma_snapshot.items()
                },
                psi=named_trainable_parameter_tensors(_as_torch_module(model)),
            ),
            parameters=_single_model_fedmatch_parameters(parameters),
            enable_inter_client_consistency=enable_inter_client_consistency,
        )
        return result.total_loss

    run_optimizer_loss_step(
        optimizer=optimizer,
        trainable_parameters=trainable_model_parameters(_as_torch_module(model)),
        max_grad_norm=max_grad_norm,
        compute_loss=compute_loss,
    )
    assert result is not None
    return result


def _apply_physical_fedmatch_unsupervised_step(
    *,
    model: ptm.PartitionedTrainableTextClassifier,
    sigma_partition_snapshot: Mapping[str, Tensor],
    unsupervised_partition: str,
    unlabeled_batch: Mapping[str, Tensor],
    parameters: FedMatchLocalObjectiveParameters,
    optimizer: torch.optim.Optimizer,
    helper_weak_probability_provider: HelperWeakProbabilityProvider | None,
    enable_inter_client_consistency: bool,
    max_grad_norm: float,
) -> FedMatchTensorLocalObjectiveResult:
    result: FedMatchTensorLocalObjectiveResult | None = None

    def compute_loss() -> Tensor:
        nonlocal result

        weak_logits = model.forward_partition(
            unsupervised_partition,
            input_ids=unlabeled_batch["weak_input_ids"],
            attention_mask=unlabeled_batch["weak_attention_mask"],
        )
        confidence_mask = _fedmatch_confidence_mask(
            weak_logits=weak_logits,
            confidence_threshold=parameters.confidence_threshold,
        )
        selected_strong_logits = _forward_selected_physical_strong_view(
            model=model,
            partition_name=unsupervised_partition,
            unlabeled_batch=unlabeled_batch,
            confidence_mask=confidence_mask,
            reference_logits=weak_logits,
        )
        selected_helper_probabilities = _compute_selected_helper_probabilities(
            helper_weak_probability_provider=helper_weak_probability_provider,
            unlabeled_batch=unlabeled_batch,
            confidence_mask=confidence_mask,
        )
        result = compute_fedmatch_unsupervised_loss(
            weak_logits=weak_logits,
            selected_strong_logits=selected_strong_logits,
            selected_helper_weak_probabilities=selected_helper_probabilities,
            parameter_partitions=FedMatchParameterPartitions(
                sigma={
                    key: value.to(device=weak_logits.device, dtype=weak_logits.dtype)
                    for key, value in sigma_partition_snapshot.items()
                },
                psi=model.partition_parameter_tensors(unsupervised_partition),
            ),
            parameters=parameters,
            enable_inter_client_consistency=enable_inter_client_consistency,
        )
        return result.total_loss

    run_optimizer_loss_step(
        optimizer=optimizer,
        trainable_parameters=model.partition_parameters(unsupervised_partition),
        max_grad_norm=max_grad_norm,
        compute_loss=compute_loss,
    )
    assert result is not None
    return result


def _single_model_fedmatch_parameters(
    parameters: FedMatchLocalObjectiveParameters,
) -> FedMatchLocalObjectiveParameters:
    """LoRA slice는 sigma/psi 별도 변수 없이 단일 tensor를 순차 업데이트한다.

    원본 FedMatch의 L1/L2는 별도 psi 변수에 거는 regularizer다. 여기서 같은
    trainable tensor 전체를 psi로 보면 confident sample이 0개인 round에서도
    classifier head와 LoRA parameter가 0으로 수축한다.
    """

    if parameters.lambda_l1 == 0.0 and parameters.lambda_l2 == 0.0:
        return parameters
    return FedMatchLocalObjectiveParameters(
        confidence_threshold=parameters.confidence_threshold,
        lambda_s=parameters.lambda_s,
        lambda_i=parameters.lambda_i,
        lambda_a=parameters.lambda_a,
        lambda_l2=0.0,
        lambda_l1=0.0,
    )


def _compute_selected_helper_probabilities(
    *,
    helper_weak_probability_provider: HelperWeakProbabilityProvider | None,
    unlabeled_batch: Mapping[str, Tensor],
    confidence_mask: Tensor,
) -> Tensor | Sequence[Tensor] | None:
    if helper_weak_probability_provider is None:
        return None
    if int(confidence_mask.sum().item()) == 0:
        return None
    return helper_weak_probability_provider(
        unlabeled_batch=_select_weak_view_batch(
            unlabeled_batch=unlabeled_batch,
            confidence_mask=confidence_mask,
        ),
    )


def _select_weak_view_batch(
    *,
    unlabeled_batch: Mapping[str, Tensor],
    confidence_mask: Tensor,
) -> dict[str, Tensor]:
    expected_rows = int(confidence_mask.shape[0])
    selected_batch: dict[str, Tensor] = {}
    for key, value in unlabeled_batch.items():
        if (
            isinstance(value, Tensor)
            and value.ndim > 0
            and value.shape[0] == expected_rows
        ):
            selected_batch[key] = value[confidence_mask]
        elif isinstance(value, Tensor):
            selected_batch[key] = value
    return selected_batch


def _fedmatch_confidence_mask(
    *,
    weak_logits: Tensor,
    confidence_threshold: float,
) -> Tensor:
    weak_probabilities = torch.softmax(weak_logits.detach(), dim=-1)
    return torch.max(weak_probabilities, dim=-1).values >= confidence_threshold


def _forward_selected_strong_view(
    *,
    model: TextBatchClassifier,
    unlabeled_batch: Mapping[str, Tensor],
    confidence_mask: Tensor,
    reference_logits: Tensor,
) -> Tensor:
    selected_count = int(confidence_mask.sum().item())
    if selected_count == 0:
        return reference_logits.new_empty((0, int(reference_logits.shape[1])))
    return model(
        input_ids=unlabeled_batch["strong_input_ids"][confidence_mask],
        attention_mask=unlabeled_batch["strong_attention_mask"][confidence_mask],
    )


def _forward_selected_physical_strong_view(
    *,
    model: ptm.PartitionedTrainableTextClassifier,
    partition_name: str,
    unlabeled_batch: Mapping[str, Tensor],
    confidence_mask: Tensor,
    reference_logits: Tensor,
) -> Tensor:
    selected_count = int(confidence_mask.sum().item())
    if selected_count == 0:
        return reference_logits.new_empty((0, int(reference_logits.shape[1])))
    return model.forward_partition(
        partition_name,
        input_ids=unlabeled_batch["strong_input_ids"][confidence_mask],
        attention_mask=unlabeled_batch["strong_attention_mask"][confidence_mask],
    )


def _apply_query_ssl_psi_step(
    *,
    model: TextBatchClassifier,
    unlabeled_batch: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    algorithm: QuerySslAlgorithm,
    max_grad_norm: float,
) -> FedMatchTensorLocalObjectiveResult:
    step_result: QuerySslStepResult | None = None

    def compute_loss() -> Tensor:
        nonlocal step_result

        step_result = algorithm.compute_step(
            model=model,
            labeled_batch=None,
            unlabeled_batch=dict(unlabeled_batch),
        )
        return step_result.total_loss

    run_optimizer_loss_step(
        optimizer=optimizer,
        trainable_parameters=trainable_model_parameters(_as_torch_module(model)),
        max_grad_norm=max_grad_norm,
        compute_loss=compute_loss,
    )
    assert step_result is not None
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


def _as_torch_module(model: TextBatchClassifier) -> nn.Module:
    return model  # type: ignore[return-value]


def _train_if_module(model: ptm.PartitionedTrainableTextClassifier) -> None:
    if isinstance(model, nn.Module):
        model.train()


def _build_partition_optimizer(
    *,
    model: ptm.PartitionedTrainableTextClassifier,
    partition_name: str,
    learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    adapter_params: list[nn.Parameter] = []
    classifier_params: list[nn.Parameter] = []
    for name, parameter in model.partition_parameter_tensors(partition_name).items():
        if not parameter.requires_grad:
            continue
        if name.startswith("classifier."):
            classifier_params.append(parameter)
            continue
        adapter_params.append(parameter)
    if not adapter_params and not classifier_params:
        raise ValueError(
            f"physical partition {partition_name!r} has no trainable parameters."
        )
    return torch.optim.AdamW(
        [
            {
                "params": adapter_params,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": classifier_params,
                "lr": classifier_learning_rate,
                "weight_decay": weight_decay,
            },
        ]
    )


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

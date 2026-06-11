"""FedMatch local objective metadata, pure helpers, and tensor loss core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from methods.federated_ssl.fedmatch.original_spec import (
    FEDMATCH_ORIGINAL_METHOD_DEFAULTS,
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
    resolve_original_scenario_spec,
)
from methods.federated_ssl.hooks.local_objective import (
    FederatedSslLocalObjectiveSpec,
    PartitionedObjectiveParameterTensors,
    TensorLocalObjectiveResult,
)
from methods.ssl.hooks.consistency import consistency_cross_entropy_loss
from methods.ssl.hooks.masking import build_fixed_threshold_mask

FEDMATCH_LOCAL_OBJECTIVE_NAME = "fedmatch_sigma_psi_local_objective"
FEDMATCH_CONFIDENCE_FILTER = "confidence_filter"
FEDMATCH_SUPERVISED_CE = "supervised_cross_entropy"
FEDMATCH_INTER_CLIENT_KL = "inter_client_consistency_kl"
FEDMATCH_AGREEMENT_PSEUDO_LABEL_CE = "agreement_pseudo_label_cross_entropy"
FEDMATCH_PSI_L1_REGULARIZATION = "psi_l1_regularization"
FEDMATCH_SIGMA_PSI_L2_REGULARIZATION = "sigma_psi_l2_regularization"


@dataclass(frozen=True, slots=True)
class FedMatchLossComponentSpec:
    """원본 FedMatch loss 항을 TraceMind partition 의미로 고정한다."""

    name: str
    coefficient_name: str | None
    updated_partition: str | None
    original_source: str
    trace_mapping: str


FEDMATCH_LOSS_COMPONENTS = (
    FedMatchLossComponentSpec(
        name=FEDMATCH_SUPERVISED_CE,
        coefficient_name="lambda_s",
        updated_partition="sigma",
        original_source="models/fedmatch/client.py::loss_fn_s",
        trace_mapping=("labeled rows로 CE를 계산하고 sigma partition만 업데이트한다."),
    ),
    FedMatchLossComponentSpec(
        name=FEDMATCH_CONFIDENCE_FILTER,
        coefficient_name="confidence_threshold",
        updated_partition=None,
        original_source="models/fedmatch/client.py::loss_fn_u",
        trace_mapping=("weak prediction confidence가 threshold 이상인 row만 쓴다."),
    ),
    FedMatchLossComponentSpec(
        name=FEDMATCH_INTER_CLIENT_KL,
        coefficient_name="lambda_i",
        updated_partition="psi",
        original_source="models/fedmatch/client.py::loss_fn_u",
        trace_mapping=("helper/local weak prediction KL을 psi objective에 더한다."),
    ),
    FedMatchLossComponentSpec(
        name=FEDMATCH_AGREEMENT_PSEUDO_LABEL_CE,
        coefficient_name="lambda_a",
        updated_partition="psi",
        original_source="models/fedmatch/client.py::agreement_based_labeling",
        trace_mapping=(
            "local/helper vote pseudo-label을 strong view CE target으로 쓴다."
        ),
    ),
    FedMatchLossComponentSpec(
        name=FEDMATCH_PSI_L1_REGULARIZATION,
        coefficient_name="lambda_l1",
        updated_partition="psi",
        original_source="models/fedmatch/client.py::loss_fn_u",
        trace_mapping="psi partition에 L1 sparse regularization을 적용한다.",
    ),
    FedMatchLossComponentSpec(
        name=FEDMATCH_SIGMA_PSI_L2_REGULARIZATION,
        coefficient_name="lambda_l2",
        updated_partition="psi",
        original_source="models/fedmatch/client.py::loss_fn_u",
        trace_mapping="sigma와 psi 차이를 L2 regularization으로 묶는다.",
    ),
)


@dataclass(frozen=True, slots=True)
class FedMatchLocalObjectiveParameters:
    """FedMatch tensor loss가 쓰는 원본 hyperparameter subset."""

    confidence_threshold: float
    lambda_s: float
    lambda_i: float
    lambda_a: float
    lambda_l2: float
    lambda_l1: float

    def __post_init__(self) -> None:
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        for name in ("lambda_s", "lambda_i", "lambda_a", "lambda_l2", "lambda_l1"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must not be negative.")

    @classmethod
    def from_mapping(
        cls,
        parameters: Mapping[str, object],
    ) -> FedMatchLocalObjectiveParameters:
        """effective parameter mapping에서 tensor objective 값을 읽는다."""

        return cls(
            confidence_threshold=float(parameters["confidence_threshold"]),
            lambda_s=float(parameters["lambda_s"]),
            lambda_i=float(parameters["lambda_i"]),
            lambda_a=float(parameters["lambda_a"]),
            lambda_l2=float(parameters["lambda_l2"]),
            lambda_l1=float(parameters["lambda_l1"]),
        )

    @classmethod
    def from_original_scenario(
        cls,
        *,
        scenario_name: str = FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
    ) -> FedMatchLocalObjectiveParameters:
        """원본 scenario 기본값으로 tensor objective parameter를 만든다."""

        scenario = resolve_original_scenario_spec(scenario_name)
        return cls(
            confidence_threshold=FEDMATCH_ORIGINAL_METHOD_DEFAULTS.confidence_threshold,
            lambda_s=scenario.lambda_s,
            lambda_i=scenario.lambda_i,
            lambda_a=scenario.lambda_a,
            lambda_l2=scenario.lambda_l2,
            lambda_l1=scenario.lambda_l1,
        )


@dataclass(frozen=True, slots=True)
class FedMatchParameterPartitions:
    """FedMatch sigma/psi logical partition tensor 묶음."""

    sigma: Mapping[str, Tensor]
    psi: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        sigma_keys = set(self.sigma)
        psi_keys = set(self.psi)
        if sigma_keys != psi_keys:
            missing_sigma = sorted(psi_keys - sigma_keys)
            missing_psi = sorted(sigma_keys - psi_keys)
            raise ValueError(
                "sigma and psi partitions must contain the same parameter keys: "
                f"missing_sigma={missing_sigma}, missing_psi={missing_psi}"
            )
        for key in sorted(sigma_keys):
            if self.sigma[key].shape != self.psi[key].shape:
                raise ValueError(
                    "sigma and psi partition tensors must have matching shapes for "
                    f"{key!r}."
                )


@dataclass(frozen=True, slots=True)
class FedMatchTensorLocalObjectiveResult:
    """FedMatch tensor step의 loss, partition routing, diagnostics."""

    total_loss: Tensor
    partition_losses: Mapping[str, Tensor]
    loss_components: Mapping[str, Tensor]
    metrics: Mapping[str, Tensor]
    debug_tensors: Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class FedMatchPartitionedTensorObjective:
    """partitioned runtime에 주입되는 FedMatch tensor objective adapter."""

    parameters: FedMatchLocalObjectiveParameters
    omit_regularization_for_single_trainable_model: bool = False

    def compute_supervised_loss(
        self,
        *,
        labeled_logits: Tensor,
        labels: Tensor,
    ) -> TensorLocalObjectiveResult:
        result = compute_fedmatch_supervised_loss(
            labeled_logits=labeled_logits,
            labels=labels,
            parameters=self.parameters,
        )
        return _to_generic_objective_result(result)

    def build_confidence_mask(
        self,
        *,
        weak_logits: Tensor,
    ) -> Tensor:
        weak_probabilities = torch.softmax(weak_logits.detach(), dim=-1)
        return _build_confidence_mask_from_probabilities(
            probabilities=weak_probabilities,
            confidence_threshold=self.parameters.confidence_threshold,
        )

    def compute_unsupervised_loss(
        self,
        *,
        weak_logits: Tensor,
        selected_strong_logits: Tensor,
        parameter_tensors: PartitionedObjectiveParameterTensors,
        selected_helper_weak_probabilities: Tensor | Sequence[Tensor] | None = None,
        enable_inter_client_consistency: bool = True,
    ) -> TensorLocalObjectiveResult:
        result = compute_fedmatch_unsupervised_loss(
            weak_logits=weak_logits,
            selected_strong_logits=selected_strong_logits,
            selected_helper_weak_probabilities=selected_helper_weak_probabilities,
            parameter_partitions=FedMatchParameterPartitions(
                sigma=parameter_tensors.reference,
                psi=parameter_tensors.trainable,
            ),
            parameters=_single_model_parameters(self.parameters)
            if self.omit_regularization_for_single_trainable_model
            else self.parameters,
            enable_inter_client_consistency=enable_inter_client_consistency,
        )
        return _to_generic_objective_result(result)


local_objective_spec = FederatedSslLocalObjectiveSpec(
    objective_name=FEDMATCH_LOCAL_OBJECTIVE_NAME,
    required_batch_views=("weak_text", "strong_text"),
    metric_prefix="fedmatch_local",
    parameters={
        "supervised_partition": "sigma",
        "unsupervised_partition": "psi",
        "agreement_loss": "helper_consistency",
        "loss_components": tuple(
            component.name for component in FEDMATCH_LOSS_COMPONENTS
        ),
    },
)


def build_fedmatch_partitioned_tensor_objective(
    parameters: FedMatchLocalObjectiveParameters,
    *,
    omit_regularization_for_single_trainable_model: bool = False,
) -> FedMatchPartitionedTensorObjective:
    """update-family partitioned runtime에 넘길 FedMatch objective를 만든다."""

    return FedMatchPartitionedTensorObjective(
        parameters=parameters,
        omit_regularization_for_single_trainable_model=(
            omit_regularization_for_single_trainable_model
        ),
    )


def _to_generic_objective_result(
    result: FedMatchTensorLocalObjectiveResult,
) -> TensorLocalObjectiveResult:
    return TensorLocalObjectiveResult(
        total_loss=result.total_loss,
        partition_losses=result.partition_losses,
        loss_components=result.loss_components,
        metrics=result.metrics,
        debug_tensors=result.debug_tensors,
    )


def _single_model_parameters(
    parameters: FedMatchLocalObjectiveParameters,
) -> FedMatchLocalObjectiveParameters:
    """단일 tensor sequential runtime에서는 psi regularizer를 생략한다."""

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


def fedmatch_loss_weights(
    *,
    scenario_name: str = FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
) -> dict[str, float]:
    scenario = resolve_original_scenario_spec(scenario_name)
    return {
        "lambda_s": scenario.lambda_s,
        "lambda_i": scenario.lambda_i,
        "lambda_a": scenario.lambda_a,
        "lambda_l2": scenario.lambda_l2,
        "lambda_l1": scenario.lambda_l1,
    }


def compute_fedmatch_supervised_loss(
    *,
    labeled_logits: Tensor,
    labels: Tensor,
    parameters: FedMatchLocalObjectiveParameters,
) -> FedMatchTensorLocalObjectiveResult:
    """원본 `loss_fn_s`처럼 CE를 sigma partition에 라우팅한다."""

    _validate_logits_2d(labeled_logits, tensor_name="labeled_logits")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D class-index tensor.")
    if labels.shape[0] != labeled_logits.shape[0]:
        raise ValueError("labels row count must match labeled_logits.")

    supervised_loss = F.cross_entropy(labeled_logits, labels, reduction="mean")
    weighted_loss = supervised_loss * parameters.lambda_s
    labeled_count = labeled_logits.new_tensor(float(labeled_logits.shape[0]))
    return FedMatchTensorLocalObjectiveResult(
        total_loss=weighted_loss,
        partition_losses={"sigma": weighted_loss},
        loss_components={FEDMATCH_SUPERVISED_CE: weighted_loss},
        metrics={"labeled_count": labeled_count},
        debug_tensors={},
    )


def compute_fedmatch_unsupervised_loss(
    *,
    weak_logits: Tensor,
    selected_strong_logits: Tensor,
    parameter_partitions: FedMatchParameterPartitions,
    parameters: FedMatchLocalObjectiveParameters,
    selected_helper_weak_probabilities: Tensor | Sequence[Tensor] | None = None,
    enable_inter_client_consistency: bool = True,
) -> FedMatchTensorLocalObjectiveResult:
    """원본 `loss_fn_u`처럼 confident subset loss를 계산한다."""

    _validate_logits_2d(weak_logits, tensor_name="weak_logits")

    weak_probabilities = torch.softmax(weak_logits, dim=-1)
    weak_probabilities_for_selection = weak_probabilities.detach()
    confidence_mask = _build_confidence_mask_from_probabilities(
        probabilities=weak_probabilities_for_selection,
        confidence_threshold=parameters.confidence_threshold,
    )
    selected_count = int(confidence_mask.sum().item())
    selected_weak_probabilities = weak_probabilities[confidence_mask]
    selected_weak_probabilities_for_labels = weak_probabilities_for_selection[
        confidence_mask
    ]
    selected_helper_probabilities = _normalize_selected_helper_probability_tensor(
        selected_helper_weak_probabilities=selected_helper_weak_probabilities,
        expected_row_count=selected_count,
        expected_class_count=weak_logits.shape[1],
        reference=weak_logits,
    )
    _validate_selected_strong_logits(
        selected_strong_logits=selected_strong_logits,
        selected_count=selected_count,
        class_count=weak_logits.shape[1],
    )

    agreement_loss, pseudo_labels = _compute_agreement_pseudo_label_loss(
        selected_strong_logits=selected_strong_logits,
        selected_weak_probabilities=selected_weak_probabilities_for_labels,
        selected_helper_probabilities=selected_helper_probabilities,
        lambda_a=parameters.lambda_a,
    )
    inter_client_loss = _compute_inter_client_consistency_loss(
        selected_weak_probabilities=selected_weak_probabilities,
        selected_helper_probabilities=selected_helper_probabilities,
        lambda_i=parameters.lambda_i,
        enabled=enable_inter_client_consistency,
    )
    l1_loss, l2_loss = _compute_sigma_psi_regularization_losses(
        parameter_partitions=parameter_partitions,
        lambda_l1=parameters.lambda_l1,
        lambda_l2=parameters.lambda_l2,
        reference=weak_logits,
    )
    total_loss = agreement_loss + inter_client_loss + l1_loss + l2_loss
    helper_count = (
        0
        if selected_helper_probabilities is None
        else int(selected_helper_probabilities.shape[0])
    )
    return FedMatchTensorLocalObjectiveResult(
        total_loss=total_loss,
        partition_losses={"psi": total_loss},
        loss_components={
            FEDMATCH_INTER_CLIENT_KL: inter_client_loss,
            FEDMATCH_AGREEMENT_PSEUDO_LABEL_CE: agreement_loss,
            FEDMATCH_PSI_L1_REGULARIZATION: l1_loss,
            FEDMATCH_SIGMA_PSI_L2_REGULARIZATION: l2_loss,
        },
        metrics={
            "confident_count": weak_logits.new_tensor(float(selected_count)),
            "util_ratio": confidence_mask.float().mean(),
            "helper_count": weak_logits.new_tensor(float(helper_count)),
        },
        debug_tensors={
            "confidence_mask": confidence_mask,
            "agreement_pseudo_labels": pseudo_labels,
        },
    )


def select_confident_prediction_indices(
    probabilities: Sequence[Sequence[float]],
    *,
    confidence_threshold: float,
) -> tuple[int, ...]:
    """원본 confidence filter를 framework 독립 함수로 보존한다."""

    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    return tuple(
        row_index
        for row_index, row in enumerate(probabilities)
        if _max_probability(row) >= confidence_threshold
    )


def agreement_pseudo_label_indices(
    client_probabilities: Sequence[Sequence[float]],
    helper_probabilities_by_helper: Sequence[Sequence[Sequence[float]]] | None = None,
) -> tuple[int, ...]:
    """FedMatch agreement-based pseudo labeling의 argmax vote를 보존한다."""

    class_count = _validate_probability_rows(client_probabilities)
    helpers = tuple(helper_probabilities_by_helper or ())
    for helper_probabilities in helpers:
        helper_class_count = _validate_probability_rows(helper_probabilities)
        if len(helper_probabilities) != len(client_probabilities):
            raise ValueError("helper probability row count must match client rows.")
        if helper_class_count != class_count:
            raise ValueError("helper class count must match client probability rows.")

    labels: list[int] = []
    for row_index, row in enumerate(client_probabilities):
        votes = [_argmax(row)]
        votes.extend(_argmax(helper[row_index]) for helper in helpers)
        labels.append(_majority_vote(votes, class_count=class_count))
    return tuple(labels)


def _max_probability(row: Sequence[float]) -> float:
    _validate_probability_row(row)
    return max(float(value) for value in row)


def _argmax(row: Sequence[float]) -> int:
    _validate_probability_row(row)
    best_index = 0
    best_value = float(row[0])
    for index, value in enumerate(row[1:], start=1):
        current = float(value)
        if current > best_value:
            best_index = index
            best_value = current
    return best_index


def _majority_vote(votes: Sequence[int], *, class_count: int) -> int:
    counts = [0 for _ in range(class_count)]
    for vote in votes:
        if vote < 0 or vote >= class_count:
            raise ValueError("vote index is outside class_count.")
        counts[vote] += 1
    return _argmax(counts)


def _validate_probability_rows(rows: Sequence[Sequence[float]]) -> int:
    if not rows:
        raise ValueError("probabilities must not be empty.")
    class_count = len(rows[0])
    for row in rows:
        _validate_probability_row(row)
        if len(row) != class_count:
            raise ValueError("all probability rows must have the same class count.")
    return class_count


def _validate_probability_row(row: Sequence[float]) -> None:
    if not row:
        raise ValueError("probability row must not be empty.")


def _validate_logits_2d(logits: Tensor, *, tensor_name: str) -> None:
    if logits.ndim != 2:
        raise ValueError(f"{tensor_name} must be a 2D [batch, class] tensor.")
    if logits.shape[0] == 0:
        raise ValueError(f"{tensor_name} batch dimension must not be empty.")
    if logits.shape[1] == 0:
        raise ValueError(f"{tensor_name} class dimension must not be empty.")


def _validate_selected_strong_logits(
    *,
    selected_strong_logits: Tensor,
    selected_count: int,
    class_count: int,
) -> None:
    if selected_strong_logits.ndim != 2:
        raise ValueError(
            "selected_strong_logits must be a 2D [selected, class] tensor."
        )
    if selected_strong_logits.shape[0] != selected_count:
        raise ValueError("selected_strong_logits row count must match confident_count.")
    if selected_strong_logits.shape[1] != class_count:
        raise ValueError("selected_strong_logits class count must match weak_logits.")


def _build_confidence_mask_from_probabilities(
    *,
    probabilities: Tensor,
    confidence_threshold: float,
) -> Tensor:
    return build_fixed_threshold_mask(
        probs_x_ulb_w=probabilities,
        p_cutoff=confidence_threshold,
    ).bool()


def _normalize_selected_helper_probability_tensor(
    *,
    selected_helper_weak_probabilities: Tensor | Sequence[Tensor] | None,
    expected_row_count: int,
    expected_class_count: int,
    reference: Tensor,
) -> Tensor | None:
    if selected_helper_weak_probabilities is None:
        return None
    if isinstance(selected_helper_weak_probabilities, Tensor):
        helper_tensor = selected_helper_weak_probabilities
        if helper_tensor.ndim == 2:
            helper_tensor = helper_tensor.unsqueeze(0)
    else:
        helper_values = tuple(selected_helper_weak_probabilities)
        if not helper_values:
            return None
        helper_tensor = torch.stack(helper_values, dim=0)

    if helper_tensor.ndim != 3:
        raise ValueError(
            "selected_helper_weak_probabilities must have shape "
            "[helper, selected, class]."
        )
    if helper_tensor.shape[1] != expected_row_count:
        raise ValueError(
            "selected helper probability row count must match confident_count."
        )
    if helper_tensor.shape[2] != expected_class_count:
        raise ValueError("helper probability class count must match weak_logits.")
    return helper_tensor.to(device=reference.device, dtype=reference.dtype)


def _compute_agreement_pseudo_label_loss(
    *,
    selected_strong_logits: Tensor,
    selected_weak_probabilities: Tensor,
    selected_helper_probabilities: Tensor | None,
    lambda_a: float,
) -> tuple[Tensor, Tensor]:
    if selected_weak_probabilities.shape[0] == 0:
        empty_labels = torch.empty(
            0,
            device=selected_strong_logits.device,
            dtype=torch.long,
        )
        return selected_strong_logits.new_zeros(()), empty_labels

    pseudo_labels = _compute_agreement_pseudo_labels(
        selected_weak_probabilities=selected_weak_probabilities,
        selected_helper_probabilities=selected_helper_probabilities,
    )
    loss = consistency_cross_entropy_loss(
        logits=selected_strong_logits,
        targets=pseudo_labels,
    )
    return loss * lambda_a, pseudo_labels


def _compute_agreement_pseudo_labels(
    *,
    selected_weak_probabilities: Tensor,
    selected_helper_probabilities: Tensor | None,
) -> Tensor:
    class_count = int(selected_weak_probabilities.shape[1])
    local_votes = torch.argmax(selected_weak_probabilities, dim=-1).unsqueeze(0)
    if selected_helper_probabilities is None:
        votes = local_votes
    else:
        helper_votes = torch.argmax(selected_helper_probabilities, dim=-1)
        votes = torch.cat((local_votes, helper_votes), dim=0)
    vote_counts = F.one_hot(votes.T, num_classes=class_count).sum(dim=1)
    return torch.argmax(vote_counts, dim=-1)


def _compute_inter_client_consistency_loss(
    *,
    selected_weak_probabilities: Tensor,
    selected_helper_probabilities: Tensor | None,
    lambda_i: float,
    enabled: bool,
) -> Tensor:
    if (
        not enabled
        or selected_helper_probabilities is None
        or selected_weak_probabilities.shape[0] == 0
    ):
        return selected_weak_probabilities.new_zeros(())

    helper_count = selected_helper_probabilities.shape[0]
    clamp_min = torch.finfo(selected_weak_probabilities.dtype).eps
    local_log_probabilities = torch.log(
        selected_weak_probabilities.clamp_min(clamp_min)
    )
    expanded_local = local_log_probabilities.unsqueeze(0).expand(
        helper_count,
        -1,
        -1,
    )
    kl_loss = F.kl_div(
        expanded_local.reshape(-1, expanded_local.shape[-1]),
        selected_helper_probabilities.reshape(
            -1,
            selected_helper_probabilities.shape[-1],
        ),
        reduction="batchmean",
    )
    return kl_loss * lambda_i


def _compute_sigma_psi_regularization_losses(
    *,
    parameter_partitions: FedMatchParameterPartitions,
    lambda_l1: float,
    lambda_l2: float,
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    l1_loss = reference.new_zeros(())
    l2_loss = reference.new_zeros(())
    for key in sorted(parameter_partitions.psi):
        psi_tensor = parameter_partitions.psi[key].to(
            device=reference.device,
            dtype=reference.dtype,
        )
        sigma_tensor = parameter_partitions.sigma[key].to(
            device=reference.device,
            dtype=reference.dtype,
        )
        l1_loss = l1_loss + torch.sum(torch.abs(psi_tensor))
        l2_loss = l2_loss + torch.sum(torch.square(sigma_tensor - psi_tensor))
    return l1_loss * lambda_l1, l2_loss * lambda_l2

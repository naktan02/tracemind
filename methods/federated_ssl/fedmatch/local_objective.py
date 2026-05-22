"""FedMatch local objective metadata and pure decision helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from methods.federated_ssl.fedmatch.original_spec import (
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
    resolve_original_scenario_spec,
)
from methods.federated_ssl.local_objective import FederatedSslLocalObjectiveSpec

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
        trace_mapping="labeled rows로 CE를 계산하고 sigma partition만 업데이트한다.",
    ),
    FedMatchLossComponentSpec(
        name=FEDMATCH_CONFIDENCE_FILTER,
        coefficient_name="confidence_threshold",
        updated_partition=None,
        original_source="models/fedmatch/client.py::loss_fn_u",
        trace_mapping=(
            "weak prediction의 max probability가 threshold 이상인 unlabeled row만 쓴다."
        ),
    ),
    FedMatchLossComponentSpec(
        name=FEDMATCH_INTER_CLIENT_KL,
        coefficient_name="lambda_i",
        updated_partition="psi",
        original_source="models/fedmatch/client.py::loss_fn_u",
        trace_mapping=(
            "helper prediction과 local weak prediction의 KL을 psi objective에 더한다."
        ),
    ),
    FedMatchLossComponentSpec(
        name=FEDMATCH_AGREEMENT_PSEUDO_LABEL_CE,
        coefficient_name="lambda_a",
        updated_partition="psi",
        original_source="models/fedmatch/client.py::agreement_based_labeling",
        trace_mapping=(
            "local/helper argmax vote pseudo-label을 strong view CE target으로 쓴다."
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


def select_confident_prediction_indices(
    probabilities: Sequence[Sequence[float]],
    *,
    confidence_threshold: float,
) -> tuple[int, ...]:
    """원본 `np.max(y_pred) >= confidence` filter를 framework 독립 함수로 보존한다."""

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

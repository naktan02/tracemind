"""LoRA-classifier partitioned local training budget 해석 helper."""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil

from methods.adaptation.query_classifier_adaptation.local_training_budget import (
    LOCAL_BUDGET_POLICY_ITERATION_CAPPED,
    LOCAL_BUDGET_POLICY_LABELED_ANCHORED,
    LOCAL_BUDGET_POLICY_ORIGINAL_METHOD,
    QuerySslLocalStepPlan,
    build_labeled_anchored_query_ssl_batch_plan,
    build_query_ssl_local_step_plan,
)
from shared.src.contracts.training_contracts import TrainingTask


def normalize_partitioned_local_budget_policy(policy_name: str | None) -> str:
    """partitioned FL SSL local budget policy 이름을 정규화한다."""

    normalized = (policy_name or LOCAL_BUDGET_POLICY_ITERATION_CAPPED).strip().lower()
    normalized = normalized.replace("-", "_")
    if normalized == LOCAL_BUDGET_POLICY_LABELED_ANCHORED:
        return LOCAL_BUDGET_POLICY_ORIGINAL_METHOD
    if normalized not in {
        LOCAL_BUDGET_POLICY_ITERATION_CAPPED,
        LOCAL_BUDGET_POLICY_ORIGINAL_METHOD,
    }:
        raise ValueError(
            "Partitioned local_budget_policy must be one of "
            f"{LOCAL_BUDGET_POLICY_ITERATION_CAPPED!r}, "
            f"{LOCAL_BUDGET_POLICY_ORIGINAL_METHOD!r}."
        )
    return normalized


def resolve_partitioned_local_budget(
    *,
    policy_name: str,
    labeled_count: int,
    unlabeled_count: int,
    training_task: TrainingTask,
    configured_unlabeled_batch_size: int | None,
    effective_parameters: Mapping[str, object],
    uses_labeled_batches: bool,
) -> tuple[int, int, QuerySslLocalStepPlan]:
    """partitioned local trainer가 사용할 batch와 step plan을 만든다."""

    if policy_name == LOCAL_BUDGET_POLICY_ORIGINAL_METHOD:
        return _resolve_original_method_budget(
            labeled_count=labeled_count,
            unlabeled_count=unlabeled_count,
            effective_parameters=effective_parameters,
            uses_labeled_batches=uses_labeled_batches,
        )

    labeled_batch_size = int(training_task.batch_size)
    resolved_unlabeled_batch_size = (
        configured_unlabeled_batch_size or labeled_batch_size
    )
    labeled_loader_steps = ceil(labeled_count / labeled_batch_size)
    unlabeled_loader_steps = ceil(unlabeled_count / resolved_unlabeled_batch_size)
    return (
        labeled_batch_size if uses_labeled_batches else 0,
        resolved_unlabeled_batch_size,
        build_query_ssl_local_step_plan(
            labeled_loader_steps=labeled_loader_steps,
            unlabeled_loader_steps=unlabeled_loader_steps,
            uses_labeled_batches=uses_labeled_batches,
            local_epochs=int(training_task.local_epochs),
            max_steps=int(training_task.max_steps),
        ),
    )


def _resolve_original_method_budget(
    *,
    labeled_count: int,
    unlabeled_count: int,
    effective_parameters: Mapping[str, object],
    uses_labeled_batches: bool,
) -> tuple[int, int, QuerySslLocalStepPlan]:
    batch_size = _required_positive_int(effective_parameters, "client_batch_size")
    local_epochs = _required_positive_int(effective_parameters, "client_epochs")
    if not uses_labeled_batches:
        unlabeled_loader_steps = ceil(unlabeled_count / batch_size)
        total_steps = local_epochs * unlabeled_loader_steps
        return (
            0,
            batch_size,
            build_query_ssl_local_step_plan(
                labeled_loader_steps=0,
                unlabeled_loader_steps=unlabeled_loader_steps,
                uses_labeled_batches=False,
                local_epochs=local_epochs,
                max_steps=total_steps,
            ),
        )

    local_budget = build_labeled_anchored_query_ssl_batch_plan(
        labeled_count=labeled_count,
        unlabeled_count=unlabeled_count,
        labeled_batch_size=batch_size,
        local_epochs=local_epochs,
    )
    return (
        local_budget.labeled_batch_size,
        local_budget.unlabeled_batch_size,
        local_budget.step_plan,
    )


def _required_positive_int(
    source: Mapping[str, object],
    key: str,
) -> int:
    if key not in source:
        raise ValueError(f"partitioned effective_parameters missing {key!r}.")
    value = int(source[key])
    if value <= 0:
        raise ValueError(f"partitioned effective_parameters.{key} must be positive.")
    return value

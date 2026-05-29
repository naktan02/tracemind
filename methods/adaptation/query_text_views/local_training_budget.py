"""Query SSL local training budget 계산 helper."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

LOCAL_BUDGET_POLICY_ITERATION_CAPPED = "iteration_capped"
LOCAL_BUDGET_POLICY_ORIGINAL_METHOD = "original_method"
LOCAL_BUDGET_POLICY_LABELED_ANCHORED = "labeled_anchored"


@dataclass(frozen=True, slots=True)
class QuerySslLocalStepPlan:
    """한 client/process가 수행할 실제 optimizer step 계획."""

    labeled_loader_steps: int
    unlabeled_loader_steps: int
    full_epoch_steps: int
    local_epochs: int
    max_steps: int
    total_steps: int


@dataclass(frozen=True, slots=True)
class QuerySslLabeledAnchoredBatchPlan:
    """labeled loader step 수에 맞춰 unlabeled batch size를 정하는 local budget."""

    step_plan: QuerySslLocalStepPlan
    labeled_batch_size: int
    unlabeled_batch_size: int
    labeled_count: int
    unlabeled_count: int


def build_query_ssl_local_step_plan(
    *,
    labeled_loader_steps: int,
    unlabeled_loader_steps: int,
    uses_labeled_batches: bool,
    local_epochs: int,
    max_steps: int,
) -> QuerySslLocalStepPlan:
    """epoch/batch/max-step 설정을 실제 Query SSL optimizer step 수로 해석한다."""

    if labeled_loader_steps < 0 or unlabeled_loader_steps < 0:
        raise ValueError("loader step counts must not be negative.")
    if unlabeled_loader_steps == 0:
        raise ValueError("Query SSL unlabeled loader must not be empty.")
    if uses_labeled_batches and labeled_loader_steps == 0:
        raise ValueError(
            "Query SSL labeled loader must not be empty when the algorithm "
            "uses labeled batches."
        )
    if local_epochs <= 0:
        raise ValueError("local_epochs must be positive.")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    full_epoch_steps = (
        max(labeled_loader_steps, unlabeled_loader_steps)
        if uses_labeled_batches
        else unlabeled_loader_steps
    )
    epoch_budget_steps = int(local_epochs) * full_epoch_steps
    return QuerySslLocalStepPlan(
        labeled_loader_steps=int(labeled_loader_steps),
        unlabeled_loader_steps=int(unlabeled_loader_steps),
        full_epoch_steps=int(full_epoch_steps),
        local_epochs=int(local_epochs),
        max_steps=int(max_steps),
        total_steps=min(int(max_steps), epoch_budget_steps),
    )


def build_labeled_anchored_query_ssl_batch_plan(
    *,
    labeled_count: int,
    unlabeled_count: int,
    labeled_batch_size: int,
    local_epochs: int,
) -> QuerySslLabeledAnchoredBatchPlan:
    """labeled batch 개수를 epoch step으로 삼고 unlabeled 전체를 같은 epoch에 맞춘다.

    labeled data가 local supervised anchor이고 unlabeled pool도 한 local epoch 안에서
    거의 한 번 지나가야 하는 SSL method가 공유할 수 있는 budget primitive다.
    """

    if labeled_count <= 0:
        raise ValueError("labeled-anchored SSL budget requires labeled rows.")
    if unlabeled_count <= 0:
        raise ValueError("labeled-anchored SSL budget requires unlabeled rows.")
    if labeled_batch_size <= 0:
        raise ValueError("labeled_batch_size must be positive.")
    if local_epochs <= 0:
        raise ValueError("local_epochs must be positive.")

    steps_per_epoch = max(1, round(int(labeled_count) / int(labeled_batch_size)))
    unlabeled_batch_size = max(1, ceil(int(unlabeled_count) / steps_per_epoch))
    total_steps = int(local_epochs) * steps_per_epoch
    step_plan = QuerySslLocalStepPlan(
        labeled_loader_steps=steps_per_epoch,
        unlabeled_loader_steps=ceil(int(unlabeled_count) / unlabeled_batch_size),
        full_epoch_steps=steps_per_epoch,
        local_epochs=int(local_epochs),
        max_steps=total_steps,
        total_steps=total_steps,
    )
    return QuerySslLabeledAnchoredBatchPlan(
        step_plan=step_plan,
        labeled_batch_size=int(labeled_batch_size),
        unlabeled_batch_size=unlabeled_batch_size,
        labeled_count=int(labeled_count),
        unlabeled_count=int(unlabeled_count),
    )

"""Query SSL local training budget 계산 helper."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuerySslLocalStepPlan:
    """한 client/process가 수행할 실제 optimizer step 계획."""

    labeled_loader_steps: int
    unlabeled_loader_steps: int
    full_epoch_steps: int
    local_epochs: int
    max_steps: int
    total_steps: int


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

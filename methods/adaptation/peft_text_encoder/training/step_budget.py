"""PEFT text encoder/head epoch/step budget compatibility surface."""

from __future__ import annotations

from methods.adaptation.common import step_budget as _common_step_budget

EpochDistributedStepBudget = _common_step_budget.EpochDistributedStepBudget
resolve_epoch_distributed_step_budget = (
    _common_step_budget.resolve_epoch_distributed_step_budget
)
remaining_effective_epochs = _common_step_budget.remaining_effective_epochs

"""LoRA-classifier epoch/step budget 해석 primitive."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True, slots=True)
class EpochDistributedStepBudget:
    """총 optimizer step을 epoch surface에 분배한 실행 계획."""

    total_train_steps: int
    steps_per_epoch_budget: int
    effective_epochs: int

    def remaining_epoch_steps(self, completed_steps: int) -> int:
        """현재 epoch에서 더 수행할 step 수를 반환한다."""

        return min(
            self.steps_per_epoch_budget,
            self.total_train_steps - int(completed_steps),
        )


def resolve_epoch_distributed_step_budget(
    *,
    epochs: int,
    full_epoch_steps: int,
    max_train_steps: int | None,
    invalid_max_steps_message: str,
) -> EpochDistributedStepBudget:
    """기존 LoRA loop의 max-step/epoch 분배 공식을 보존한다."""

    if max_train_steps is not None and max_train_steps <= 0:
        raise ValueError(invalid_max_steps_message)

    total_train_steps = (
        int(max_train_steps)
        if max_train_steps is not None
        else int(epochs) * int(full_epoch_steps)
    )
    effective_epoch_count = max(1, int(epochs))
    steps_per_epoch_budget = max(1, ceil(total_train_steps / effective_epoch_count))
    effective_epochs = min(effective_epoch_count, total_train_steps)
    return EpochDistributedStepBudget(
        total_train_steps=int(total_train_steps),
        steps_per_epoch_budget=int(steps_per_epoch_budget),
        effective_epochs=int(effective_epochs),
    )


def remaining_effective_epochs(
    *,
    epochs: int,
    remaining_steps: int,
    steps_per_epoch_budget: int,
) -> int:
    """resume 후 남은 step 수를 epoch 수로 환산한다."""

    return min(
        max(1, int(epochs)),
        max(1, ceil(int(remaining_steps) / int(steps_per_epoch_budget))),
    )

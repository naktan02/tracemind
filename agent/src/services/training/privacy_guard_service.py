"""로컬 학습 update의 privacy/safety 보호 계층."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)
from shared.src.domain.entities.training.training_task import TrainingTask
from shared.src.domain.entities.training.vector_adapter_delta import (
    VectorAdapterDelta,
)


@dataclass(slots=True)
class PrivacyProtectedUpdate:
    """privacy guard 적용 결과."""

    update: SharedAdapterUpdate
    clipped: bool = False
    dp_applied: bool = False


class SharedAdapterPrivacyGuard(Protocol):
    """Shared adapter update에 clipping/DP를 적용하는 인터페이스."""

    def protect(
        self,
        *,
        update: SharedAdapterUpdate,
        training_task: TrainingTask,
    ) -> PrivacyProtectedUpdate:
        """update에 privacy/safety 보호 계층을 적용한다."""


@dataclass(slots=True)
class DiagonalScaleClipOnlyPrivacyGuard:
    """현재 diagonal scale update에 clip만 적용하는 기본 privacy guard."""

    def protect(
        self,
        *,
        update: SharedAdapterUpdate,
        training_task: TrainingTask,
    ) -> PrivacyProtectedUpdate:
        if not isinstance(update, VectorAdapterDelta):
            raise TypeError(
                "DiagonalScaleClipOnlyPrivacyGuard expects VectorAdapterDelta, "
                f"got {type(update)!r}."
            )

        clip_norm = training_task.gradient_clip_norm
        if clip_norm is None:
            return PrivacyProtectedUpdate(update=update)

        current_norm = update.l2_norm()
        if current_norm == 0.0 or current_norm <= clip_norm:
            return PrivacyProtectedUpdate(update=update)

        scale = clip_norm / current_norm
        clipped = replace(
            update,
            dimension_deltas=[value * scale for value in update.dimension_deltas],
        )
        return PrivacyProtectedUpdate(update=clipped, clipped=True)

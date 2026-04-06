"""로컬 학습 update의 privacy/safety 보호 계층."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from shared.src.contracts.adapter_contracts import VectorAdapterDelta
from shared.src.contracts.training_contracts import TrainingTask
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)


@dataclass(slots=True)
class PrivacyProtectedUpdate:
    """privacy guard 적용 결과."""

    update: SharedAdapterUpdate
    clipped: bool = False
    dp_applied: bool = False


class SharedAdapterPrivacyGuard(Protocol):
    """Shared adapter update에 clipping/DP를 적용하는 인터페이스."""

    guard_name: str

    def protect(
        self,
        *,
        update: SharedAdapterUpdate,
        training_task: TrainingTask,
    ) -> PrivacyProtectedUpdate:
        """update에 privacy/safety 보호 계층을 적용한다."""


@dataclass(slots=True)
class NoOpSharedAdapterPrivacyGuard:
    """privacy/safety 처리를 적용하지 않는 no-op guard."""

    guard_name: str = "noop"

    def protect(
        self,
        *,
        update: SharedAdapterUpdate,
        training_task: TrainingTask,
    ) -> PrivacyProtectedUpdate:
        del training_task
        return PrivacyProtectedUpdate(update=update)


@dataclass(slots=True)
class DiagonalScaleClipOnlyPrivacyGuard:
    """현재 diagonal scale update에 clip만 적용하는 기본 privacy guard."""

    guard_name: str = "diagonal_scale_clip_only"

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
        clipped = update.model_copy(
            update={
                "dimension_deltas": [value * scale for value in update.dimension_deltas]
            }
        )
        return PrivacyProtectedUpdate(update=clipped, clipped=True)


def build_shared_adapter_privacy_guard(
    guard_name: str,
) -> SharedAdapterPrivacyGuard:
    """guard 이름으로 privacy guard를 생성한다."""

    normalized_name = guard_name.strip().lower()
    if normalized_name == "diagonal_scale_clip_only":
        return DiagonalScaleClipOnlyPrivacyGuard()
    if normalized_name == "noop":
        return NoOpSharedAdapterPrivacyGuard()
    raise ValueError(f"Unsupported privacy guard: {guard_name}.")

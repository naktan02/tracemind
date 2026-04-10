"""로컬 학습 update의 privacy/safety 보호 계층."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from shared.src.config.adapter_family_metadata import (
    CLASSIFIER_HEAD_FAMILY_METADATA,
    DIAGONAL_SCALE_FAMILY_METADATA,
)
from shared.src.contracts.adapter_contracts import (
    ClassifierHeadDelta,
    VectorAdapterDelta,
)
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
    supported_adapter_kinds: tuple[str, ...]

    def protect(
        self,
        *,
        update: SharedAdapterUpdate,
        training_task: TrainingTask,
    ) -> PrivacyProtectedUpdate:
        """update에 privacy/safety 보호 계층을 적용한다."""


PrivacyGuardFactory = Callable[[], SharedAdapterPrivacyGuard]


@dataclass(slots=True)
class NoOpSharedAdapterPrivacyGuard:
    """privacy/safety 처리를 적용하지 않는 no-op guard."""

    guard_name: str = "noop"
    supported_adapter_kinds: tuple[str, ...] = ("*",)

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
    supported_adapter_kinds: tuple[str, ...] = (
        DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,
    )

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


@dataclass(slots=True)
class ClassifierHeadClipOnlyPrivacyGuard:
    """Classifier-head update에 clip만 적용하는 privacy guard."""

    guard_name: str = "classifier_head_clip_only"
    supported_adapter_kinds: tuple[str, ...] = (
        CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,
    )

    def protect(
        self,
        *,
        update: SharedAdapterUpdate,
        training_task: TrainingTask,
    ) -> PrivacyProtectedUpdate:
        if not isinstance(update, ClassifierHeadDelta):
            raise TypeError(
                "ClassifierHeadClipOnlyPrivacyGuard expects ClassifierHeadDelta, "
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
                "label_weight_deltas": {
                    label: [value * scale for value in deltas]
                    for label, deltas in update.label_weight_deltas.items()
                },
                "label_bias_deltas": {
                    label: value * scale
                    for label, value in update.label_bias_deltas.items()
                },
            }
        )
        return PrivacyProtectedUpdate(update=clipped, clipped=True)


_PRIVACY_GUARD_REGISTRY: dict[str, PrivacyGuardFactory] = {}


def register_shared_adapter_privacy_guard(
    *guard_names: str,
    factory: PrivacyGuardFactory,
) -> None:
    """얇은 wiring registry에 privacy guard를 등록한다."""
    for guard_name in guard_names:
        _PRIVACY_GUARD_REGISTRY[guard_name.strip().lower()] = factory


def build_shared_adapter_privacy_guard(
    guard_name: str,
) -> SharedAdapterPrivacyGuard:
    """guard 이름으로 privacy guard를 생성한다."""

    normalized_name = guard_name.strip().lower()
    factory = _PRIVACY_GUARD_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory()
    raise ValueError(f"Unsupported privacy guard: {guard_name}.")


register_shared_adapter_privacy_guard(
    "diagonal_scale_clip_only",
    factory=DiagonalScaleClipOnlyPrivacyGuard,
)
register_shared_adapter_privacy_guard(
    "classifier_head_clip_only",
    factory=ClassifierHeadClipOnlyPrivacyGuard,
)
register_shared_adapter_privacy_guard(
    "noop",
    factory=NoOpSharedAdapterPrivacyGuard,
)

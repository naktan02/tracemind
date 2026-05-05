"""로컬 학습 update의 privacy/safety 보호 계층."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from shared.src.config.adapter_family_metadata import (
    CLASSIFIER_HEAD_FAMILY_METADATA,
    DIAGONAL_SCALE_FAMILY_METADATA,
)
from shared.src.config.local_training_registry_catalog import (
    CLASSIFIER_HEAD_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY,
    DIAGONAL_SCALE_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY,
    NOOP_PRIVACY_GUARD_CATALOG_ENTRY,
)
from shared.src.config.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
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


_PRIVACY_GUARD_REGISTRY: dict[
    str,
    tuple[PrivacyGuardFactory, RegistryCatalogEntry],
] = {}


def register_shared_adapter_privacy_guard(
    *guard_names: str,
    factory: PrivacyGuardFactory,
    catalog_entry: RegistryCatalogEntry,
) -> None:
    """얇은 wiring registry에 privacy guard를 등록한다."""
    registered_guard = (factory, catalog_entry)
    for guard_name in guard_names:
        _PRIVACY_GUARD_REGISTRY[guard_name.strip().lower()] = registered_guard


def build_shared_adapter_privacy_guard(
    guard_name: str,
) -> SharedAdapterPrivacyGuard:
    """guard 이름으로 privacy guard를 생성한다."""

    normalized_name = guard_name.strip().lower()
    registered_guard = _PRIVACY_GUARD_REGISTRY.get(normalized_name)
    if registered_guard is not None:
        factory, _catalog_entry = registered_guard
        return factory()
    raise ValueError(f"Unsupported privacy guard: {guard_name}.")


def list_registered_shared_adapter_privacy_guard_names() -> tuple[str, ...]:
    """등록된 privacy guard 이름을 정렬된 tuple로 반환한다."""

    return tuple(sorted(_PRIVACY_GUARD_REGISTRY))


def list_shared_adapter_privacy_guard_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 privacy guard catalog entry를 canonical item 기준으로 반환한다."""

    return dedupe_registry_catalog_entries(
        catalog_entry for _factory, catalog_entry in _PRIVACY_GUARD_REGISTRY.values()
    )


register_shared_adapter_privacy_guard(
    "diagonal_scale_clip_only",
    factory=DiagonalScaleClipOnlyPrivacyGuard,
    catalog_entry=DIAGONAL_SCALE_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY,
)
register_shared_adapter_privacy_guard(
    "classifier_head_clip_only",
    factory=ClassifierHeadClipOnlyPrivacyGuard,
    catalog_entry=CLASSIFIER_HEAD_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY,
)
register_shared_adapter_privacy_guard(
    "noop",
    factory=NoOpSharedAdapterPrivacyGuard,
    catalog_entry=NOOP_PRIVACY_GUARD_CATALOG_ENTRY,
)

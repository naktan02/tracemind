"""Diagonal-scale clip-only privacy guard."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    DIAGONAL_SCALE_ADAPTER_KIND,
)
from shared.src.contracts.adapter_contracts import VectorAdapterDelta
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingTask
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from .base import PrivacyProtectedUpdate
from .registry import register_shared_adapter_privacy_guard

DIAGONAL_SCALE_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="diagonal_scale_clip_only",
    display_name="diagonal_scale_clip_only",
    implementation_module=(
        "agent.src.services.training.execution.privacy_guards.diagonal_scale_clip_only"
    ),
    core_method_name="diagonal_scale_clip_only",
    family_name="privacy_guard",
    supported_adapter_kinds=(DIAGONAL_SCALE_ADAPTER_KIND,),
)


@dataclass(slots=True)
class DiagonalScaleClipOnlyPrivacyGuard:
    """현재 diagonal scale update에 clip만 적용하는 기본 privacy guard."""

    guard_name: str = "diagonal_scale_clip_only"
    supported_adapter_kinds: tuple[str, ...] = (DIAGONAL_SCALE_ADAPTER_KIND,)

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


@register_shared_adapter_privacy_guard(
    "diagonal_scale_clip_only",
    catalog_entry=DIAGONAL_SCALE_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY,
)
def build_diagonal_scale_clip_only_privacy_guard() -> DiagonalScaleClipOnlyPrivacyGuard:
    """registry용 diagonal-scale clip-only privacy guard factory."""

    return DiagonalScaleClipOnlyPrivacyGuard()

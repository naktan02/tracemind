"""Clip-only privacy guards for shared adapter updates."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.contracts.adapter_contract_families.classifier_head import (
    CLASSIFIER_HEAD_ADAPTER_KIND,
    ClassifierHeadDelta,
)
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingTask
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from .base import PrivacyProtectedUpdate
from .registry import register_shared_adapter_privacy_guard

CLASSIFIER_HEAD_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="classifier_head_clip_only",
    display_name="classifier_head_clip_only",
    implementation_module="methods.adaptation.privacy_guards.clip_only",
    core_method_name="classifier_head_clip_only",
    family_name="privacy_guard",
    supported_adapter_kinds=(CLASSIFIER_HEAD_ADAPTER_KIND,),
)


@dataclass(frozen=True, slots=True)
class ClipOnlyPrivacyGuard:
    """update L2 norm을 TrainingTask gradient_clip_norm 이하로 제한한다."""

    guard_name: str
    supported_adapter_kinds: tuple[str, ...]
    update_type: type[SharedAdapterUpdate]

    def protect(
        self,
        *,
        update: SharedAdapterUpdate,
        training_task: TrainingTask,
    ) -> PrivacyProtectedUpdate:
        if not isinstance(update, self.update_type):
            raise TypeError(
                f"{self.guard_name} expects {self.update_type.__name__}, "
                f"got {type(update)!r}."
            )

        clip_norm = training_task.gradient_clip_norm
        if clip_norm is None:
            return PrivacyProtectedUpdate(update=update)

        current_norm = update.l2_norm()
        if current_norm == 0.0 or current_norm <= clip_norm:
            return PrivacyProtectedUpdate(update=update)

        scale = clip_norm / current_norm
        return PrivacyProtectedUpdate(
            update=_scale_update_delta(update, scale=scale),
            clipped=True,
        )


@register_shared_adapter_privacy_guard(
    "classifier_head_clip_only",
    catalog_entry=CLASSIFIER_HEAD_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY,
)
def build_classifier_head_clip_only_privacy_guard() -> ClipOnlyPrivacyGuard:
    """registry용 head clip-only privacy guard factory."""

    return ClipOnlyPrivacyGuard(
        guard_name="classifier_head_clip_only",
        supported_adapter_kinds=(CLASSIFIER_HEAD_ADAPTER_KIND,),
        update_type=ClassifierHeadDelta,
    )


def _scale_update_delta(
    update: SharedAdapterUpdate,
    *,
    scale: float,
) -> SharedAdapterUpdate:
    if isinstance(update, ClassifierHeadDelta):
        return update.model_copy(
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
    raise TypeError(f"Unsupported clip-only update type: {type(update)!r}.")

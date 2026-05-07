"""Classifier-head clip-only privacy guard."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.contracts.adapter_contracts import ClassifierHeadDelta
from shared.src.contracts.adapter_family_metadata import CLASSIFIER_HEAD_FAMILY_METADATA
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
    implementation_module=(
        "agent.src.services.training.execution.privacy_guards."
        "classifier_head_clip_only"
    ),
    core_method_name="classifier_head_clip_only",
    family_name="privacy_guard",
    supported_adapter_kinds=(CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,),
)


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


@register_shared_adapter_privacy_guard(
    "classifier_head_clip_only",
    catalog_entry=CLASSIFIER_HEAD_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY,
)
def build_classifier_head_clip_only_privacy_guard() -> (
    ClassifierHeadClipOnlyPrivacyGuard
):
    """registry용 classifier-head clip-only privacy guard factory."""

    return ClassifierHeadClipOnlyPrivacyGuard()

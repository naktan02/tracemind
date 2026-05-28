"""No-op shared adapter privacy guard."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingTask
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from .base import PrivacyProtectedUpdate
from .registry import register_shared_adapter_privacy_guard

NOOP_PRIVACY_GUARD_NAME = "noop"

NOOP_PRIVACY_GUARD_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name=NOOP_PRIVACY_GUARD_NAME,
    display_name=NOOP_PRIVACY_GUARD_NAME,
    implementation_module="methods.adaptation.privacy_guards.noop",
    core_method_name=NOOP_PRIVACY_GUARD_NAME,
    family_name="privacy_guard",
    supported_adapter_kinds=("*",),
)


@dataclass(slots=True)
class NoOpSharedAdapterPrivacyGuard:
    """privacy/safety 처리를 적용하지 않는 no-op guard."""

    guard_name: str = NOOP_PRIVACY_GUARD_NAME
    supported_adapter_kinds: tuple[str, ...] = ("*",)

    def protect(
        self,
        *,
        update: SharedAdapterUpdate,
        training_task: TrainingTask,
    ) -> PrivacyProtectedUpdate:
        del training_task
        return PrivacyProtectedUpdate(update=update)


@register_shared_adapter_privacy_guard(
    NOOP_PRIVACY_GUARD_NAME,
    catalog_entry=NOOP_PRIVACY_GUARD_CATALOG_ENTRY,
)
def build_noop_privacy_guard() -> NoOpSharedAdapterPrivacyGuard:
    """registry용 no-op privacy guard factory."""

    return NoOpSharedAdapterPrivacyGuard()

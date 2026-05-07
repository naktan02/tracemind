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

NOOP_PRIVACY_GUARD_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="noop",
    display_name="noop",
    implementation_module=(
        "agent.src.services.training.execution.privacy_guards.noop"
    ),
    core_method_name="noop",
    family_name="privacy_guard",
    supported_adapter_kinds=("*",),
)


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


@register_shared_adapter_privacy_guard(
    "noop",
    catalog_entry=NOOP_PRIVACY_GUARD_CATALOG_ENTRY,
)
def build_noop_privacy_guard() -> NoOpSharedAdapterPrivacyGuard:
    """registry용 no-op privacy guard factory."""

    return NoOpSharedAdapterPrivacyGuard()

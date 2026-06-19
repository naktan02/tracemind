"""Agent active shared adapter state runtime service."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.src.infrastructure.repositories.shared_adapter_state_repository import (
    SharedAdapterStateRepository,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterStatePayload,
)
from shared.src.contracts.model_contracts import ModelManifestPayload


@dataclass(slots=True)
class SharedAdapterRuntimeService:
    """현재 활성 shared adapter state와 manifest를 제공한다."""

    repository: SharedAdapterStateRepository = field(
        default_factory=SharedAdapterStateRepository
    )

    def get_active_state(self) -> SharedAdapterStatePayload:
        return self.repository.load_active_state()

    def get_active_manifest(self) -> ModelManifestPayload:
        return self.repository.load_active_manifest()

"""Agent active PrototypePack runtime service."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.src.infrastructure.repositories.prototype_pack_repository import (
    PrototypePackRepository,
)
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    extract_category_centroids,
)


@dataclass(slots=True)
class PrototypeRuntimeService:
    """현재 활성 prototype pack과 centroid 매핑을 제공한다."""

    repository: PrototypePackRepository = field(default_factory=PrototypePackRepository)

    def get_active_pack(self) -> PrototypePackPayload:
        active_pointer = self.repository.load_active_pointer()
        if active_pointer is None:
            raise FileNotFoundError("No active prototype pack is cached on the agent.")
        return self.repository.load_pack(active_pointer.prototype_version)

    def get_active_centroids(self) -> dict[str, list[float]]:
        return extract_category_centroids(self.get_active_pack())

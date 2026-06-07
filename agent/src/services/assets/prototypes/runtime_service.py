"""Agent active PrototypePack runtime service."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.src.infrastructure.repositories.prototype_pack_repository import (
    PrototypePackRepository,
)
from methods.prototype.projections import (
    project_category_centroids_by_largest_cluster,
    require_single_category_centroids,
)
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    extract_category_prototypes,
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

    def get_active_prototypes(self) -> dict[str, tuple[list[float], ...]]:
        return extract_category_prototypes(self.get_active_pack())

    def get_scoring_assets(self) -> dict[str, tuple[list[float], ...]]:
        """prototype scorer용 asset provider 호환 표면."""
        return self.get_active_prototypes()

    def get_active_single_centroids(self) -> dict[str, list[float]]:
        return require_single_category_centroids(self.get_active_pack())

    def get_active_projected_centroids(self) -> dict[str, list[float]]:
        """multi-prototype pack도 읽을 수 있는 대표 centroid view."""
        return project_category_centroids_by_largest_cluster(self.get_active_pack())

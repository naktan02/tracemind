"""Prototype build state 저장 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shared.src.contracts.prototype_build_state_contracts import (
    PrototypeBuildStatePayload,
    load_prototype_build_state_payload,
)

from ...infrastructure.repositories.prototype_build_state_repository import (
    PrototypeBuildStateRepository,
)


@dataclass(slots=True)
class PrototypeBuildStateService:
    """Prototype build state의 저장과 조회를 담당한다."""

    repository: PrototypeBuildStateRepository = field(
        default_factory=PrototypeBuildStateRepository
    )

    def publish_state(self, payload: PrototypeBuildStatePayload) -> Path:
        return self.repository.save_state(payload)

    def publish_state_file(self, source_path: Path) -> Path:
        payload = load_prototype_build_state_payload(source_path)
        return self.publish_state(payload)

    def get_state(self, prototype_version: str) -> PrototypeBuildStatePayload:
        return self.repository.load_state(prototype_version)

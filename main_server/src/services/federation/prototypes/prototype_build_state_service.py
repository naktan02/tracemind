"""Prototype build state 저장 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from main_server.src.infrastructure.repositories import (
    prototype_build_state_repository as build_state_repository_module,
)
from shared.src.contracts.prototype_build_state_contracts import (
    SinglePrototypeBuildStatePayload,
    load_prototype_build_state_payload,
)


@dataclass(slots=True)
class PrototypeBuildStateService:
    """Prototype build state의 저장과 조회를 담당한다."""

    repository: build_state_repository_module.PrototypeBuildStateRepository = field(
        default_factory=build_state_repository_module.PrototypeBuildStateRepository
    )

    def publish_state(self, payload: SinglePrototypeBuildStatePayload) -> Path:
        return self.repository.save_state(payload)

    def publish_state_file(self, source_path: Path) -> Path:
        payload = load_prototype_build_state_payload(source_path)
        return self.publish_state(payload)

    def get_state(self, prototype_version: str) -> SinglePrototypeBuildStatePayload:
        return self.repository.load_state(prototype_version)

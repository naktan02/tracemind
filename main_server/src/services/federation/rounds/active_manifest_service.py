"""서버 소유 active model manifest 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from main_server.src.infrastructure.repositories import (
    model_manifest_repository as model_manifest_repository_module,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.domain.services.clock import Clock, SystemUtcClock

ModelManifestRepository = model_manifest_repository_module.ModelManifestRepository


def _build_model_manifest_repository() -> ModelManifestRepository:
    return ModelManifestRepository()


@dataclass(slots=True)
class ActiveModelManifestService:
    """현재 서버가 배포하는 전역 model manifest를 관리한다."""

    manifest_repository: ModelManifestRepository = field(
        default_factory=_build_model_manifest_repository
    )
    clock: Clock = field(default_factory=SystemUtcClock)

    def save_and_activate(
        self,
        manifest: ModelManifest,
        *,
        activated_at: datetime | None = None,
    ) -> ModelManifest:
        """manifest를 revision별로 저장하고 current pointer를 갱신한다."""

        effective_activated_at = activated_at or self.clock.now()
        self.manifest_repository.save_model_manifest(manifest)
        self.manifest_repository.set_active(
            manifest.model_revision,
            activated_at=effective_activated_at,
        )
        return manifest

    def get_active_manifest(self) -> ModelManifest:
        """현재 active model manifest를 반환한다."""

        return self.manifest_repository.load_active_model_manifest()

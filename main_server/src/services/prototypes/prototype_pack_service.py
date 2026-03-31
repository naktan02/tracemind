"""PrototypePack 배포 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shared.src.contracts.prototype_contracts import (
    CurrentPrototypePackResponse,
    PrototypePackActivationPointer,
    PrototypePackPayload,
    load_prototype_pack_payload,
)
from main_server.src.infrastructure.repositories.prototype_pack_repository import (
    PrototypePackRepository,
)


@dataclass(slots=True)
class PrototypePackService:
    """PrototypePack의 저장, 활성화, 조회를 담당한다."""

    repository: PrototypePackRepository = field(default_factory=PrototypePackRepository)

    def publish_pack(self, payload: PrototypePackPayload) -> Path:
        return self.repository.save_pack(payload)

    def publish_pack_file(self, source_path: Path) -> Path:
        payload = load_prototype_pack_payload(source_path)
        return self.publish_pack(payload)

    def activate(self, prototype_version: str) -> PrototypePackActivationPointer:
        return self.repository.set_active(prototype_version)

    def get_pack(self, prototype_version: str) -> PrototypePackPayload:
        return self.repository.load_pack(prototype_version)

    def get_current(self) -> CurrentPrototypePackResponse:
        active = self.repository.load_active_pointer()
        if active is None:
            raise FileNotFoundError("No active prototype pack is registered.")
        return CurrentPrototypePackResponse(
            active=active,
            pack=self.repository.load_pack(active.prototype_version),
        )

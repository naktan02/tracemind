"""중앙 서버에서 PrototypePack을 내려받아 agent 캐시에 반영한다."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from urllib.parse import urljoin
from urllib.request import urlopen

from agent.src.infrastructure.repositories.prototype_pack_repository import (
    PrototypePackRepository,
)
from shared.src.contracts.prototype_contracts import (
    CurrentPrototypePackResponse,
    PrototypePackActivationPointer,
)


@dataclass(slots=True)
class PrototypeSyncService:
    """현재 활성 prototype pack을 중앙에서 가져와 로컬 캐시에 반영한다."""

    repository: PrototypePackRepository = field(default_factory=PrototypePackRepository)

    def pull_current(self, *, server_base_url: str) -> PrototypePackActivationPointer:
        endpoint = urljoin(server_base_url.rstrip("/") + "/", "api/v1/prototypes/current")
        with urlopen(endpoint) as response:
            payload = json.loads(response.read().decode("utf-8"))

        current = CurrentPrototypePackResponse.model_validate(payload)
        self.repository.save_pack(current.pack)
        return self.repository.set_active(current.pack.prototype_version)

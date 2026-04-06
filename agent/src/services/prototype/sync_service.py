"""중앙 서버에서 PrototypePack을 내려받아 agent 캐시에 반영한다."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

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
    timeout: float = 10.0
    _transport: httpx.BaseTransport | None = field(default=None, repr=False)

    def _client(self, server_base_url: str) -> httpx.Client:
        kwargs: dict[str, object] = {
            "base_url": server_base_url,
            "timeout": self.timeout,
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def pull_current(self, *, server_base_url: str) -> PrototypePackActivationPointer:
        with self._client(server_base_url) as client:
            response = client.get("/api/v1/prototypes/current")
            if response.status_code == 404:
                raise FileNotFoundError("No active prototype pack is registered.")
            response.raise_for_status()
            payload = response.json()

        current = CurrentPrototypePackResponse.model_validate(payload)
        self.repository.save_pack(current.pack)
        return self.repository.set_active(current.pack.prototype_version)

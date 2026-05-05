"""중앙 서버에서 shared adapter state를 내려받아 agent 캐시에 반영한다."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from agent.src.infrastructure.repositories.shared_adapter_state_repository import (
    SharedAdapterStateActivationPointer,
    SharedAdapterStateRepository,
)
from shared.src.contracts.adapter_contracts import CurrentSharedAdapterStatePayload


@dataclass(slots=True)
class SharedAdapterSyncService:
    """현재 활성 shared adapter state를 중앙에서 가져와 로컬 캐시에 반영한다."""

    repository: SharedAdapterStateRepository = field(
        default_factory=SharedAdapterStateRepository
    )
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

    def pull_current(
        self,
        *,
        server_base_url: str,
    ) -> SharedAdapterStateActivationPointer:
        with self._client(server_base_url) as client:
            response = client.get("/api/v1/fl/rounds/active-state/current")
            if response.status_code == 404:
                raise FileNotFoundError("No active shared adapter state is registered.")
            response.raise_for_status()
            payload = response.json()

        current = CurrentSharedAdapterStatePayload.model_validate(payload)
        return self.repository.save_current(
            manifest=current.manifest,
            state=current.state,
        )

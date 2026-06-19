"""Agent runtime profile server sync service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from agent.src.features.assets.shared_adapters.sync_service import (
    SharedAdapterSyncService,
)
from agent.src.features.runtime_profile.repository import (
    RuntimeProfileRecord,
    RuntimeProfileRepository,
)
from shared.src.contracts.agent_runtime_profile_contracts import (
    AgentRuntimeProfilePayload,
    AgentRuntimeProfileValidationRequestPayload,
    AgentRuntimeProfileValidationResponsePayload,
)


@dataclass(frozen=True, slots=True)
class RuntimeProfileSyncResult:
    """runtime profile sync 결과."""

    status: str
    profile: AgentRuntimeProfilePayload | None
    message: str


@dataclass(slots=True)
class RuntimeProfileSyncService:
    """서버 active runtime profile을 agent-local cache에 반영한다."""

    repository: RuntimeProfileRepository = field(
        default_factory=RuntimeProfileRepository
    )
    shared_adapter_sync_service: SharedAdapterSyncService = field(
        default_factory=SharedAdapterSyncService
    )
    timeout: float = 10.0
    _transport: httpx.BaseTransport | None = field(default=None, repr=False)

    def sync_current(self, *, server_base_url: str) -> RuntimeProfileSyncResult:
        active = self.repository.load_active()
        latest = self._validate_or_fetch_latest(
            server_base_url=server_base_url,
            active=active,
        )
        if latest is None and active is not None:
            self.shared_adapter_sync_service.pull_current(
                server_base_url=server_base_url
            )
            self.repository.mark_server_validated(
                profile_id=active.profile.profile_id,
                profile_revision=active.profile.profile_revision,
                payload_checksum=active.profile.payload_checksum,
                validated_at=datetime.now(tz=timezone.utc),
                server_base_url=server_base_url,
            )
            return RuntimeProfileSyncResult(
                status="up_to_date",
                profile=active.profile,
                message="Runtime profile is up to date.",
            )

        if latest is None:
            return RuntimeProfileSyncResult(
                status="not_configured",
                profile=None,
                message="No active runtime profile is configured on the server.",
            )

        self.shared_adapter_sync_service.pull_current(server_base_url=server_base_url)
        record = self.repository.save_profile(
            latest,
            source="server",
            activate=True,
            server_validated_at=datetime.now(tz=timezone.utc),
            server_base_url=server_base_url,
        )
        return RuntimeProfileSyncResult(
            status="updated",
            profile=record.profile,
            message="Runtime profile was updated from server.",
        )

    def _validate_or_fetch_latest(
        self,
        *,
        server_base_url: str,
        active: RuntimeProfileRecord | None,
    ) -> AgentRuntimeProfilePayload | None:
        with self._client(server_base_url) as client:
            if active is None:
                response = client.get("/api/v1/agent-runtime-profile/current")
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return AgentRuntimeProfilePayload.model_validate(response.json())

            response = client.post(
                "/api/v1/agent-runtime-profile/validate",
                json=AgentRuntimeProfileValidationRequestPayload(
                    profile_id=active.profile.profile_id,
                    profile_revision=active.profile.profile_revision,
                    payload_checksum=active.profile.payload_checksum,
                    model_revision=active.profile.model_revision,
                ).model_dump(mode="json"),
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            result = AgentRuntimeProfileValidationResponsePayload.model_validate(
                response.json()
            )
            return None if result.up_to_date else result.latest_profile

    def _client(self, server_base_url: str) -> httpx.Client:
        kwargs: dict[str, object] = {
            "base_url": server_base_url,
            "timeout": self.timeout,
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

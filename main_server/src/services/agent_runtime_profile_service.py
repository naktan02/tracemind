"""Server-owned agent runtime profile service."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from main_server.src.infrastructure.repositories.agent_runtime_profile_repository import (  # noqa: E501
    AgentRuntimeProfileRepository,
)
from main_server.src.services.federation.rounds.active_manifest_service import (
    ActiveModelManifestService,
)
from shared.src.contracts.agent_runtime_profile_contracts import (
    AgentRuntimeProfilePayload,
    AgentRuntimeProfileValidationRequestPayload,
    AgentRuntimeProfileValidationResponsePayload,
    make_agent_runtime_profile_payload,
)
from shared.src.domain.services.clock import Clock, SystemUtcClock

RUNTIME_PROFILE_ID_ENV = "TRACEMIND_AGENT_RUNTIME_PROFILE_ID"
RUNTIME_PROFILE_REVISION_ENV = "TRACEMIND_AGENT_RUNTIME_PROFILE_REVISION"
RUNTIME_FAMILY_ENV = "TRACEMIND_AGENT_RUNTIME_FAMILY"
ADAPTER_MECHANISM_ENV = "TRACEMIND_AGENT_RUNTIME_ADAPTER_MECHANISM"
SCORER_BACKEND_ENV = "TRACEMIND_AGENT_RUNTIME_SCORER_BACKEND"
EMBEDDING_BACKEND_ENV = "TRACEMIND_AGENT_RUNTIME_EMBEDDING_BACKEND"
EMBEDDING_MODEL_ID_ENV = "TRACEMIND_AGENT_RUNTIME_EMBEDDING_MODEL_ID"
REQUIRED_STATE_KIND_ENV = "TRACEMIND_AGENT_RUNTIME_REQUIRED_STATE_KIND"


@dataclass(slots=True)
class AgentRuntimeProfileService:
    """main_server가 agent 상시 분석 runtime profile을 제공한다."""

    repository: AgentRuntimeProfileRepository = field(
        default_factory=AgentRuntimeProfileRepository
    )
    active_manifest_service: ActiveModelManifestService = field(
        default_factory=ActiveModelManifestService
    )
    clock: Clock = field(default_factory=SystemUtcClock)
    environ: Mapping[str, str] | None = None

    def get_current_profile(self) -> AgentRuntimeProfilePayload:
        """active runtime profile을 반환한다.

        명시 저장된 profile이 없으면 서버 환경변수와 active manifest로 profile을
        구성한다. 필요한 환경값이 없으면 FileNotFoundError를 낸다.
        """

        try:
            return self.repository.load_active()
        except FileNotFoundError:
            return self._profile_from_env_and_manifest()

    def save_active_profile(
        self,
        profile: AgentRuntimeProfilePayload,
    ) -> AgentRuntimeProfilePayload:
        return self.repository.save_active(profile)

    def validate_profile(
        self,
        request: AgentRuntimeProfileValidationRequestPayload,
    ) -> AgentRuntimeProfileValidationResponsePayload:
        current = self.get_current_profile()
        up_to_date = (
            request.profile_id == current.profile_id
            and request.profile_revision == current.profile_revision
            and request.payload_checksum == current.payload_checksum
        )
        return AgentRuntimeProfileValidationResponsePayload(
            up_to_date=up_to_date,
            latest_profile=None if up_to_date else current,
        )

    def _profile_from_env_and_manifest(self) -> AgentRuntimeProfilePayload:
        environ = os.environ if self.environ is None else self.environ
        manifest = self.active_manifest_service.get_active_manifest()
        profile_id = _required_env(environ, RUNTIME_PROFILE_ID_ENV)
        profile_revision = environ.get(RUNTIME_PROFILE_REVISION_ENV, "").strip()
        return make_agent_runtime_profile_payload(
            profile_id=profile_id,
            profile_revision=profile_revision or manifest.model_revision,
            model_id=manifest.model_id,
            model_revision=manifest.model_revision,
            runtime_family=_required_env(environ, RUNTIME_FAMILY_ENV),
            adapter_mechanism=_optional_env(environ, ADAPTER_MECHANISM_ENV),
            scorer_backend_name=_required_env(environ, SCORER_BACKEND_ENV),
            embedding_backend=_required_env(environ, EMBEDDING_BACKEND_ENV),
            embedding_model_id=(
                _optional_env(environ, EMBEDDING_MODEL_ID_ENV) or manifest.model_id
            ),
            training_scope=str(manifest.training_scope),
            required_state_kind=_optional_env(environ, REQUIRED_STATE_KIND_ENV),
            updated_at=self.clock.now(),
        )


def _required_env(environ: Mapping[str, str], key: str) -> str:
    value = environ.get(key, "").strip()
    if not value:
        raise FileNotFoundError(
            f"No active agent runtime profile is configured and {key} is not set."
        )
    return value


def _optional_env(environ: Mapping[str, str], key: str) -> str | None:
    value = environ.get(key, "").strip()
    return value or None

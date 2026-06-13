"""Agent runtime profile 계약.

이 payload는 서버가 agent 상시 분석 runtime에 적용할 scorer/model profile을
내려줄 때 쓰는 canonical shape다. 허용 runtime family와 backend catalog는
각 runtime owner가 검증하고, shared contract는 shape와 checksum 규칙만 소유한다.
"""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

AGENT_RUNTIME_PROFILE_V1 = "agent_runtime_profile.v1"
AGENT_RUNTIME_PROFILE_VALIDATION_REQUEST_V1 = (
    "agent_runtime_profile_validation_request.v1"
)
AGENT_RUNTIME_PROFILE_VALIDATION_RESPONSE_V1 = (
    "agent_runtime_profile_validation_response.v1"
)

AgentRuntimeProfileSchemaVersion: TypeAlias = Literal["agent_runtime_profile.v1"]
AgentRuntimeProfileValidationRequestSchemaVersion: TypeAlias = Literal[
    "agent_runtime_profile_validation_request.v1"
]
AgentRuntimeProfileValidationResponseSchemaVersion: TypeAlias = Literal[
    "agent_runtime_profile_validation_response.v1"
]


class AgentRuntimeProfilePayload(BaseModel):
    """Agent 상시 분석 pipeline을 구성하는 서버 배포 profile.

    `profile_id`, `profile_revision`, `payload_checksum`이 최신성 비교 기준이다.
    `updated_at`은 표시, 감사, stale 정책 판단에만 쓰고 동등성 비교에는 쓰지 않는다.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: AgentRuntimeProfileSchemaVersion = AGENT_RUNTIME_PROFILE_V1
    profile_id: str = Field(min_length=1, max_length=128)
    profile_revision: str = Field(min_length=1, max_length=128)
    payload_checksum: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=256)
    model_revision: str = Field(min_length=1, max_length=128)
    runtime_family: str = Field(min_length=1, max_length=80)
    adapter_mechanism: str | None = Field(default=None, max_length=80)
    scorer_backend_name: str = Field(min_length=1, max_length=128)
    embedding_backend: str = Field(min_length=1, max_length=128)
    embedding_model_id: str = Field(min_length=1, max_length=256)
    training_scope: str = Field(min_length=1, max_length=80)
    required_state_kind: str | None = Field(default=None, max_length=128)
    updated_at: datetime

    @model_validator(mode="after")
    def _validate_checksum(self) -> "AgentRuntimeProfilePayload":
        expected = compute_agent_runtime_profile_checksum(self)
        if self.payload_checksum != expected:
            raise ValueError("payload_checksum does not match runtime profile payload.")
        return self

    def identity_matches(
        self,
        other: "AgentRuntimeProfilePayload",
    ) -> bool:
        """최신성 비교에 사용할 stable identity가 같은지 확인한다."""

        return (
            self.profile_id == other.profile_id
            and self.profile_revision == other.profile_revision
            and self.payload_checksum == other.payload_checksum
        )


class AgentRuntimeProfileValidationRequestPayload(BaseModel):
    """Agent가 가진 active profile이 서버 최신 profile과 같은지 확인하는 요청."""

    model_config = ConfigDict(extra="forbid")

    schema_version: AgentRuntimeProfileValidationRequestSchemaVersion = (
        AGENT_RUNTIME_PROFILE_VALIDATION_REQUEST_V1
    )
    profile_id: str | None = Field(default=None, max_length=128)
    profile_revision: str | None = Field(default=None, max_length=128)
    payload_checksum: str | None = Field(default=None, max_length=128)
    model_revision: str | None = Field(default=None, max_length=128)


class AgentRuntimeProfileValidationResponsePayload(BaseModel):
    """서버 active runtime profile 최신성 확인 결과."""

    model_config = ConfigDict(extra="forbid")

    schema_version: AgentRuntimeProfileValidationResponseSchemaVersion = (
        AGENT_RUNTIME_PROFILE_VALIDATION_RESPONSE_V1
    )
    up_to_date: bool
    latest_profile: AgentRuntimeProfilePayload | None = None

    @model_validator(mode="after")
    def _validate_latest_profile(
        self,
    ) -> "AgentRuntimeProfileValidationResponsePayload":
        if not self.up_to_date and self.latest_profile is None:
            raise ValueError("latest_profile is required when up_to_date is false.")
        return self


def make_agent_runtime_profile_payload(
    *,
    profile_id: str,
    profile_revision: str,
    model_id: str,
    model_revision: str,
    runtime_family: str,
    adapter_mechanism: str | None,
    scorer_backend_name: str,
    embedding_backend: str,
    embedding_model_id: str,
    training_scope: str,
    required_state_kind: str | None,
    updated_at: datetime,
) -> AgentRuntimeProfilePayload:
    """checksum을 계산해 AgentRuntimeProfilePayload를 만든다."""

    draft = {
        "schema_version": AGENT_RUNTIME_PROFILE_V1,
        "profile_id": profile_id,
        "profile_revision": profile_revision,
        "model_id": model_id,
        "model_revision": model_revision,
        "runtime_family": runtime_family,
        "adapter_mechanism": adapter_mechanism,
        "scorer_backend_name": scorer_backend_name,
        "embedding_backend": embedding_backend,
        "embedding_model_id": embedding_model_id,
        "training_scope": training_scope,
        "required_state_kind": required_state_kind,
    }
    return AgentRuntimeProfilePayload(
        **draft,
        updated_at=updated_at,
        payload_checksum=_checksum_mapping(draft),
    )


def compute_agent_runtime_profile_checksum(
    payload: AgentRuntimeProfilePayload,
) -> str:
    """비교용 canonical JSON checksum을 계산한다.

    `updated_at`은 저장/표시용 metadata라서 checksum identity에 포함하지 않는다.
    """

    values = payload.model_dump(
        mode="json",
        exclude={"payload_checksum", "updated_at"},
    )
    return _checksum_mapping(values)


def _checksum_mapping(values: dict[str, object]) -> str:
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(encoded.encode("utf-8")).hexdigest()

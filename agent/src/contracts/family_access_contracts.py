"""가족용 확장 프로그램의 setup/auth 경계 계약."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated

FAMILY_SETUP_STATUS_V1 = "family_setup_status.v1"
FAMILY_SETUP_RESPONSE_V1 = "family_setup_response.v1"
FAMILY_UNLOCK_RESPONSE_V1 = "family_unlock_response.v1"

FamilySetupStatusSchemaVersion: TypeAlias = Literal["family_setup_status.v1"]
FamilySetupResponseSchemaVersion: TypeAlias = Literal["family_setup_response.v1"]
FamilyUnlockResponseSchemaVersion: TypeAlias = Literal["family_unlock_response.v1"]

PinCode: TypeAlias = Annotated[str, StringConstraints(pattern=r"^\d{4,6}$")]


class FamilyAccessRole(StrEnum):
    """가족용 확장 프로그램의 역할 구분."""

    CHILD = "child"
    PARENT = "parent"


class FamilyAccessMode(StrEnum):
    """현재 family extension이 지원하는 agent 연결 모드."""

    THIS_DEVICE_ONLY = "this_device_only"


class FamilySetupStatusPayload(BaseModel):
    """최초 setup이 끝났는지와 어떤 역할이 준비됐는지 반환한다."""

    model_config = ConfigDict(extra="forbid")

    schema_version: FamilySetupStatusSchemaVersion = FAMILY_SETUP_STATUS_V1
    access_mode: FamilyAccessMode = FamilyAccessMode.THIS_DEVICE_ONLY
    is_setup_complete: bool
    configured_roles: tuple[FamilyAccessRole, ...] = Field(default_factory=tuple)


class FamilySetupRequestPayload(BaseModel):
    """최초 setup 시 child/parent PIN을 함께 설정한다."""

    model_config = ConfigDict(extra="forbid")

    child_pin: PinCode
    parent_pin: PinCode


class FamilySetupResponsePayload(BaseModel):
    """최초 setup 결과."""

    model_config = ConfigDict(extra="forbid")

    schema_version: FamilySetupResponseSchemaVersion = FAMILY_SETUP_RESPONSE_V1
    access_mode: FamilyAccessMode = FamilyAccessMode.THIS_DEVICE_ONLY
    is_setup_complete: bool
    configured_roles: tuple[FamilyAccessRole, ...] = Field(default_factory=tuple)


class FamilyUnlockRequestPayload(BaseModel):
    """role별 잠금 해제를 위한 PIN 요청."""

    model_config = ConfigDict(extra="forbid")

    role: FamilyAccessRole
    pin: PinCode


class FamilyUnlockResponsePayload(BaseModel):
    """role별 잠금 해제 결과."""

    model_config = ConfigDict(extra="forbid")

    schema_version: FamilyUnlockResponseSchemaVersion = FAMILY_UNLOCK_RESPONSE_V1
    role: FamilyAccessRole
    granted: bool
    session_token: str | None = None
    session_expires_at: datetime | None = None
    remaining_attempts: int | None = Field(default=None, ge=0)
    locked_until: datetime | None = None


def _dump_payload(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def dump_family_setup_status_payload(
    path: Path,
    payload: FamilySetupStatusPayload,
) -> None:
    _dump_payload(path, payload)


def dump_family_setup_response_payload(
    path: Path,
    payload: FamilySetupResponsePayload,
) -> None:
    _dump_payload(path, payload)


def dump_family_unlock_response_payload(
    path: Path,
    payload: FamilyUnlockResponsePayload,
) -> None:
    _dump_payload(path, payload)

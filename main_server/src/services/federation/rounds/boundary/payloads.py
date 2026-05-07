"""FL round API/persistence payload와 직렬화 유틸리티."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from main_server.src.services.federation.rounds.boundary.models import RoundStatus
from methods.federated_ssl.training_defaults import DEFAULT_TRAINING_PROFILE
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.model_contracts import ModelManifestPayload
from shared.src.contracts.training_contracts import (
    SecureAggregationConfigPayload,
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
    TrainingTaskPayload,
    TrainingUpdateEnvelopePayload,
)


class RoundPublicationPayload(BaseModel):
    """라운드 publication API/repository payload."""

    model_config = ConfigDict(extra="forbid")

    next_manifest: ModelManifestPayload
    aggregated_metrics: dict[str, float]
    update_count: int = Field(ge=1)
    finalized_at: datetime
    prototype_pack_ref: str | None = None
    prototype_build_state_ref: str | None = None
    prototype_rebuild_input_id: str | None = None


class RoundRecordPayload(BaseModel):
    """라운드 상태 저장/응답 payload."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "fl_round_record.v1"
    round_id: str
    status: RoundStatus
    active_manifest: ModelManifestPayload
    training_task: TrainingTaskPayload
    updates: list[TrainingUpdateEnvelopePayload] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    finalized_at: datetime | None = None
    publication: RoundPublicationPayload | None = None


class ActiveRoundPointerPayload(BaseModel):
    """현재 active round 포인터."""

    model_config = ConfigDict(extra="forbid")

    round_id: str
    activated_at: datetime


class ActiveModelManifestPointerPayload(BaseModel):
    """현재 active model manifest 포인터."""

    model_config = ConfigDict(extra="forbid")

    model_revision: str
    activated_at: datetime


class RoundTaskConfigPayload(BaseModel):
    """round task template API payload."""

    model_config = ConfigDict(extra="forbid")

    task_type: TrainingTaskType = TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING
    local_epochs: int = Field(default=DEFAULT_TRAINING_PROFILE.local_epochs, ge=1)
    batch_size: int = Field(default=DEFAULT_TRAINING_PROFILE.batch_size, ge=1)
    learning_rate: float = Field(
        default=DEFAULT_TRAINING_PROFILE.learning_rate,
        gt=0.0,
    )
    max_steps: int = Field(default=DEFAULT_TRAINING_PROFILE.max_steps, ge=1)
    objective_config: TrainingObjectiveConfigPayload | None = None
    selection_policy: TrainingSelectionPolicyPayload | None = None
    secure_aggregation: SecureAggregationConfigPayload | None = None
    min_required_examples: int | None = Field(
        default=DEFAULT_TRAINING_PROFILE.min_required_examples,
        ge=1,
    )
    gradient_clip_norm: float | None = Field(
        default=DEFAULT_TRAINING_PROFILE.gradient_clip_norm,
        gt=0.0,
    )
    deadline_at: datetime | None = None
    notes: str | None = None


class RoundOpenRequestPayload(RoundTaskConfigPayload):
    """round open API payload."""

    model_config = ConfigDict(extra="forbid")

    round_id: str | None = None
    task_id: str | None = None


class RoundFinalizeRequestPayload(BaseModel):
    """round finalize API payload."""

    model_config = ConfigDict(extra="forbid")

    next_prototype_version: str
    next_model_revision: str | None = None
    published_at: datetime | None = None


class RoundUpdateAcceptancePayload(BaseModel):
    """update 수락 API 응답 payload."""

    model_config = ConfigDict(extra="forbid")

    round_id: str
    update_id: str
    update_count: int = Field(ge=1)
    accepted_at: datetime
    idempotent: bool = False


def _dump_payload(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def dump_round_record_payload(path: Path, payload: RoundRecordPayload) -> None:
    """RoundRecordPayload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_round_record_payload(path: Path) -> RoundRecordPayload:
    """JSON 파일에서 RoundRecordPayload를 읽는다."""
    return RoundRecordPayload.model_validate_json(path.read_text(encoding="utf-8"))


def dump_active_round_pointer_payload(
    path: Path,
    payload: ActiveRoundPointerPayload,
) -> None:
    """active round 포인터를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_active_round_pointer_payload(path: Path) -> ActiveRoundPointerPayload:
    """JSON 파일에서 active round 포인터를 읽는다."""
    return ActiveRoundPointerPayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_active_model_manifest_pointer_payload(
    path: Path,
    payload: ActiveModelManifestPointerPayload,
) -> None:
    """active model manifest 포인터를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_active_model_manifest_pointer_payload(
    path: Path,
) -> ActiveModelManifestPointerPayload:
    """JSON 파일에서 active model manifest 포인터를 읽는다."""
    return ActiveModelManifestPointerPayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )

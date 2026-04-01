"""FL round lifecycle용 상태 모델과 직렬화 contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from shared.src.contracts.model_contracts import ModelManifestPayload
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
    TrainingTaskPayload,
    TrainingUpdateEnvelopePayload,
)
from shared.src.domain.entities.artifacts.model_manifest import ModelManifest
from shared.src.domain.entities.training.training_task import TrainingTask
from shared.src.domain.entities.training.training_task_config import (
    TrainingConfigScalar,
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)
from shared.src.domain.entities.training.training_update import TrainingUpdateEnvelope


class RoundStatus(StrEnum):
    """FL round 상태."""

    OPEN = "open"
    FINALIZED = "finalized"


@dataclass(slots=True)
class RoundPublicationSummary:
    """라운드 finalize 이후 남기는 publication 요약."""

    next_manifest: ModelManifest
    aggregated_metrics: dict[str, float]
    update_count: int
    finalized_at: datetime
    prototype_pack_ref: str | None = None
    prototype_build_state_ref: str | None = None
    prototype_rebuild_input_id: str | None = None


@dataclass(slots=True)
class RoundRecord:
    """라운드 하나의 canonical runtime 상태."""

    round_id: str
    status: RoundStatus
    active_manifest: ModelManifest
    training_task: TrainingTask
    created_at: datetime
    updated_at: datetime
    updates: tuple[TrainingUpdateEnvelope, ...] = field(default_factory=tuple)
    finalized_at: datetime | None = None
    publication: RoundPublicationSummary | None = None


@dataclass(slots=True)
class RoundOpenRequest:
    """새 round open 요청."""

    active_manifest: ModelManifest
    round_id: str | None = None
    task_id: str | None = None
    task_type: str = "pseudo_label_self_training"
    local_epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-4
    max_steps: int = 50
    objective_config: (
        TrainingObjectiveConfig | Mapping[str, TrainingConfigScalar] | None
    ) = None
    selection_policy: (
        TrainingSelectionPolicy | Mapping[str, TrainingConfigScalar] | None
    ) = None
    min_required_examples: int | None = None
    gradient_clip_norm: float | None = None
    deadline_at: datetime | None = None
    notes: str | None = None


@dataclass(slots=True)
class RoundFinalizeRequest:
    """round finalize 요청."""

    next_prototype_version: str
    next_model_revision: str | None = None
    published_at: datetime | None = None


@dataclass(slots=True)
class RoundUpdateAcceptance:
    """update 등록 성공 응답."""

    round_id: str
    update_id: str
    update_count: int
    accepted_at: datetime
    idempotent: bool = False


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


class RoundOpenRequestPayload(BaseModel):
    """round open API payload."""

    model_config = ConfigDict(extra="forbid")

    active_manifest: ModelManifestPayload
    round_id: str | None = None
    task_id: str | None = None
    task_type: str = "pseudo_label_self_training"
    local_epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    max_steps: int = Field(default=50, ge=1)
    objective_config: TrainingObjectiveConfigPayload | None = None
    selection_policy: TrainingSelectionPolicyPayload | None = None
    min_required_examples: int | None = Field(default=None, ge=1)
    gradient_clip_norm: float | None = Field(default=None, gt=0.0)
    deadline_at: datetime | None = None
    notes: str | None = None


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


def model_manifest_to_payload(manifest: ModelManifest) -> ModelManifestPayload:
    """domain manifest를 payload로 변환한다."""
    return ModelManifestPayload(
        schema_version=manifest.schema_version,
        model_id=manifest.model_id,
        model_revision=manifest.model_revision,
        published_at=manifest.published_at,
        artifact_kind=manifest.artifact_kind,
        artifact_ref=manifest.artifact_ref,
        prototype_version=manifest.prototype_version,
        training_scope=manifest.training_scope,
        training_enabled=manifest.training_enabled,
        compatible_task_types=list(manifest.compatible_task_types),
        base_model_id=manifest.base_model_id,
        base_model_revision=manifest.base_model_revision,
        translation_model_id=manifest.translation_model_id,
        translation_model_revision=manifest.translation_model_revision,
        notes=manifest.notes,
    )


def model_manifest_from_payload(payload: ModelManifestPayload) -> ModelManifest:
    """payload manifest를 domain으로 변환한다."""
    return ModelManifest(
        schema_version=payload.schema_version,
        model_id=payload.model_id,
        model_revision=payload.model_revision,
        published_at=payload.published_at,
        artifact_kind=payload.artifact_kind,
        artifact_ref=payload.artifact_ref,
        prototype_version=payload.prototype_version,
        training_scope=payload.training_scope,
        training_enabled=payload.training_enabled,
        compatible_task_types=tuple(payload.compatible_task_types),
        base_model_id=payload.base_model_id,
        base_model_revision=payload.base_model_revision,
        translation_model_id=payload.translation_model_id,
        translation_model_revision=payload.translation_model_revision,
        notes=payload.notes,
    )


def training_task_to_payload(task: TrainingTask) -> TrainingTaskPayload:
    """domain training task를 payload로 변환한다."""
    return TrainingTaskPayload(
        schema_version=task.schema_version,
        task_id=task.task_id,
        round_id=task.round_id,
        model_id=task.model_id,
        model_revision=task.model_revision,
        task_type=task.task_type,
        training_scope=task.training_scope,
        local_epochs=task.local_epochs,
        batch_size=task.batch_size,
        learning_rate=task.learning_rate,
        max_steps=task.max_steps,
        objective_config=TrainingObjectiveConfigPayload(
            loss=task.objective_config.loss,
            confidence_threshold=task.objective_config.confidence_threshold,
            margin_threshold=task.objective_config.margin_threshold,
            score_policy_name=task.objective_config.score_policy_name,
            score_top_k=task.objective_config.score_top_k,
            acceptance_policy_name=task.objective_config.acceptance_policy_name,
            privacy_guard_name=task.objective_config.privacy_guard_name,
            extras=dict(task.objective_config.extras),
        ),
        selection_policy=TrainingSelectionPolicyPayload(
            max_examples=task.selection_policy.max_examples,
            require_feedback=task.selection_policy.require_feedback,
            extras=dict(task.selection_policy.extras),
        ),
        deadline_at=task.deadline_at,
        gradient_clip_norm=task.gradient_clip_norm,
        min_required_examples=task.min_required_examples,
        secure_aggregation_required=task.secure_aggregation_required,
        notes=task.notes,
    )


def training_task_from_payload(payload: TrainingTaskPayload) -> TrainingTask:
    """payload training task를 domain으로 변환한다."""
    return TrainingTask(
        schema_version=payload.schema_version,
        task_id=payload.task_id,
        round_id=payload.round_id,
        model_id=payload.model_id,
        model_revision=payload.model_revision,
        task_type=payload.task_type,
        training_scope=payload.training_scope,
        local_epochs=payload.local_epochs,
        batch_size=payload.batch_size,
        learning_rate=payload.learning_rate,
        max_steps=payload.max_steps,
        objective_config=TrainingObjectiveConfig(
            loss=payload.objective_config.loss,
            confidence_threshold=payload.objective_config.confidence_threshold,
            margin_threshold=payload.objective_config.margin_threshold,
            score_policy_name=payload.objective_config.score_policy_name,
            score_top_k=payload.objective_config.score_top_k,
            acceptance_policy_name=payload.objective_config.acceptance_policy_name,
            privacy_guard_name=payload.objective_config.privacy_guard_name,
            extras=dict(payload.objective_config.extras),
        ),
        selection_policy=TrainingSelectionPolicy(
            max_examples=payload.selection_policy.max_examples,
            require_feedback=payload.selection_policy.require_feedback,
            extras=dict(payload.selection_policy.extras),
        ),
        deadline_at=payload.deadline_at,
        gradient_clip_norm=payload.gradient_clip_norm,
        min_required_examples=payload.min_required_examples,
        secure_aggregation_required=payload.secure_aggregation_required,
        notes=payload.notes,
    )


def training_update_to_payload(
    envelope: TrainingUpdateEnvelope,
) -> TrainingUpdateEnvelopePayload:
    """domain training update를 payload로 변환한다."""
    return TrainingUpdateEnvelopePayload(
        schema_version=envelope.schema_version,
        update_id=envelope.update_id,
        round_id=envelope.round_id,
        task_id=envelope.task_id,
        model_id=envelope.model_id,
        base_model_revision=envelope.base_model_revision,
        training_scope=envelope.training_scope,
        payload_ref=envelope.payload_ref,
        payload_format=envelope.payload_format,
        example_count=envelope.example_count,
        client_metrics=dict(envelope.client_metrics),
        created_at=envelope.created_at,
        clipped=envelope.clipped,
        dp_applied=envelope.dp_applied,
        checksum=envelope.checksum,
        notes=envelope.notes,
    )


def training_update_from_payload(
    payload: TrainingUpdateEnvelopePayload,
) -> TrainingUpdateEnvelope:
    """payload training update를 domain으로 변환한다."""
    return TrainingUpdateEnvelope(
        schema_version=payload.schema_version,
        update_id=payload.update_id,
        round_id=payload.round_id,
        task_id=payload.task_id,
        model_id=payload.model_id,
        base_model_revision=payload.base_model_revision,
        training_scope=payload.training_scope,
        payload_ref=payload.payload_ref,
        payload_format=payload.payload_format,
        example_count=payload.example_count,
        client_metrics=dict(payload.client_metrics),
        created_at=payload.created_at,
        clipped=payload.clipped,
        dp_applied=payload.dp_applied,
        checksum=payload.checksum,
        notes=payload.notes,
    )


def round_publication_to_payload(
    publication: RoundPublicationSummary,
) -> RoundPublicationPayload:
    """domain publication summary를 payload로 변환한다."""
    return RoundPublicationPayload(
        next_manifest=model_manifest_to_payload(publication.next_manifest),
        aggregated_metrics=dict(publication.aggregated_metrics),
        update_count=publication.update_count,
        finalized_at=publication.finalized_at,
        prototype_pack_ref=publication.prototype_pack_ref,
        prototype_build_state_ref=publication.prototype_build_state_ref,
        prototype_rebuild_input_id=publication.prototype_rebuild_input_id,
    )


def round_publication_from_payload(
    payload: RoundPublicationPayload,
) -> RoundPublicationSummary:
    """payload publication summary를 domain으로 변환한다."""
    return RoundPublicationSummary(
        next_manifest=model_manifest_from_payload(payload.next_manifest),
        aggregated_metrics=dict(payload.aggregated_metrics),
        update_count=payload.update_count,
        finalized_at=payload.finalized_at,
        prototype_pack_ref=payload.prototype_pack_ref,
        prototype_build_state_ref=payload.prototype_build_state_ref,
        prototype_rebuild_input_id=payload.prototype_rebuild_input_id,
    )


def round_record_to_payload(record: RoundRecord) -> RoundRecordPayload:
    """domain round record를 payload로 변환한다."""
    return RoundRecordPayload(
        round_id=record.round_id,
        status=record.status,
        active_manifest=model_manifest_to_payload(record.active_manifest),
        training_task=training_task_to_payload(record.training_task),
        updates=[training_update_to_payload(update) for update in record.updates],
        created_at=record.created_at,
        updated_at=record.updated_at,
        finalized_at=record.finalized_at,
        publication=(
            round_publication_to_payload(record.publication)
            if record.publication is not None
            else None
        ),
    )


def round_record_from_payload(payload: RoundRecordPayload) -> RoundRecord:
    """payload round record를 domain으로 변환한다."""
    return RoundRecord(
        round_id=payload.round_id,
        status=payload.status,
        active_manifest=model_manifest_from_payload(payload.active_manifest),
        training_task=training_task_from_payload(payload.training_task),
        updates=tuple(
            training_update_from_payload(update_payload)
            for update_payload in payload.updates
        ),
        created_at=payload.created_at,
        updated_at=payload.updated_at,
        finalized_at=payload.finalized_at,
        publication=(
            round_publication_from_payload(payload.publication)
            if payload.publication is not None
            else None
        ),
    )


def round_open_request_from_payload(
    payload: RoundOpenRequestPayload,
) -> RoundOpenRequest:
    """API payload를 domain open request로 변환한다."""
    return RoundOpenRequest(
        active_manifest=model_manifest_from_payload(payload.active_manifest),
        round_id=payload.round_id,
        task_id=payload.task_id,
        task_type=payload.task_type,
        local_epochs=payload.local_epochs,
        batch_size=payload.batch_size,
        learning_rate=payload.learning_rate,
        max_steps=payload.max_steps,
        objective_config=(
            TrainingObjectiveConfig(
                loss=payload.objective_config.loss,
                confidence_threshold=payload.objective_config.confidence_threshold,
                margin_threshold=payload.objective_config.margin_threshold,
                score_policy_name=payload.objective_config.score_policy_name,
                score_top_k=payload.objective_config.score_top_k,
                acceptance_policy_name=payload.objective_config.acceptance_policy_name,
                privacy_guard_name=payload.objective_config.privacy_guard_name,
                extras=dict(payload.objective_config.extras),
            )
            if payload.objective_config is not None
            else None
        ),
        selection_policy=(
            TrainingSelectionPolicy(
                max_examples=payload.selection_policy.max_examples,
                require_feedback=payload.selection_policy.require_feedback,
                extras=dict(payload.selection_policy.extras),
            )
            if payload.selection_policy is not None
            else None
        ),
        min_required_examples=payload.min_required_examples,
        gradient_clip_norm=payload.gradient_clip_norm,
        deadline_at=payload.deadline_at,
        notes=payload.notes,
    )


def round_finalize_request_from_payload(
    payload: RoundFinalizeRequestPayload,
) -> RoundFinalizeRequest:
    """API payload를 domain finalize request로 변환한다."""
    return RoundFinalizeRequest(
        next_prototype_version=payload.next_prototype_version,
        next_model_revision=payload.next_model_revision,
        published_at=payload.published_at,
    )


def round_update_acceptance_to_payload(
    acceptance: RoundUpdateAcceptance,
) -> RoundUpdateAcceptancePayload:
    """domain update acceptance를 API payload로 변환한다."""
    return RoundUpdateAcceptancePayload(
        round_id=acceptance.round_id,
        update_id=acceptance.update_id,
        update_count=acceptance.update_count,
        accepted_at=acceptance.accepted_at,
        idempotent=acceptance.idempotent,
    )

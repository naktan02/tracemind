"""Round update acceptance/idempotency 정책."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from main_server.src.services.rounds.models import RoundRecord
from shared.src.domain.entities.training.training_update import TrainingUpdateEnvelope


class RoundConflictError(ValueError):
    """현재 round 상태와 충돌하는 요청."""


class RoundValidationError(ValueError):
    """round 문맥과 맞지 않는 입력."""


class UpdateAcceptanceAction(StrEnum):
    """update acceptance 결과 종류."""

    ACCEPT = "accept"
    IDEMPOTENT = "idempotent"


@dataclass(slots=True)
class UpdateAcceptanceDecision:
    """policy가 계산한 update acceptance 결과."""

    action: UpdateAcceptanceAction
    update_envelope: TrainingUpdateEnvelope

    @property
    def is_idempotent(self) -> bool:
        return self.action == UpdateAcceptanceAction.IDEMPOTENT


class RoundUpdateAcceptancePolicy(Protocol):
    """round update acceptance/idempotency 정책 인터페이스."""

    def evaluate(
        self,
        *,
        record: RoundRecord,
        update: TrainingUpdateEnvelope,
        accepted_at: datetime,
    ) -> UpdateAcceptanceDecision:
        """update를 accept할지, idempotent로 처리할지 결정한다."""


def _normalize_update(
    update: TrainingUpdateEnvelope,
    *,
    accepted_at: datetime,
) -> TrainingUpdateEnvelope:
    if update.created_at is not None:
        return update
    return replace(update, created_at=accepted_at)


def _validate_update_context(
    *,
    record: RoundRecord,
    update: TrainingUpdateEnvelope,
) -> None:
    if update.round_id != record.round_id:
        raise RoundValidationError(
            f"Update round_id does not match target round: {update.round_id}"
        )
    if update.task_id != record.training_task.task_id:
        raise RoundValidationError(
            "Update task_id does not match the active training task: "
            f"{update.task_id}"
        )
    if update.model_id != record.active_manifest.model_id:
        raise RoundValidationError(
            "Update model_id does not match the active manifest: "
            f"{update.model_id}"
        )
    if update.base_model_revision != record.active_manifest.model_revision:
        raise RoundValidationError(
            "Update base_model_revision does not match the active manifest: "
            f"{update.base_model_revision}"
        )
    if update.training_scope != record.training_task.training_scope:
        raise RoundValidationError(
            "Update training_scope does not match the active training task: "
            f"{update.training_scope}"
        )


def _find_existing_update(
    *,
    record: RoundRecord,
    update_id: str,
) -> TrainingUpdateEnvelope | None:
    for existing in record.updates:
        if existing.update_id == update_id:
            return existing
    return None


def _idempotency_fingerprint(
    update: TrainingUpdateEnvelope,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    int,
    tuple[tuple[str, float], ...],
    bool | None,
    bool | None,
    str | None,
    str | None,
]:
    return (
        update.schema_version,
        update.update_id,
        update.round_id,
        update.task_id,
        update.model_id,
        update.base_model_revision,
        update.training_scope,
        update.example_count,
        tuple(sorted(update.client_metrics.items())),
        update.clipped,
        update.dp_applied,
        update.checksum,
        update.payload_ref,
    )


@dataclass(slots=True)
class StrictRoundUpdateAcceptancePolicy:
    """같은 update_id 재전송을 거부하는 기본 정책."""

    def evaluate(
        self,
        *,
        record: RoundRecord,
        update: TrainingUpdateEnvelope,
        accepted_at: datetime,
    ) -> UpdateAcceptanceDecision:
        _validate_update_context(record=record, update=update)
        if _find_existing_update(record=record, update_id=update.update_id) is not None:
            raise RoundConflictError(
                f"Duplicate update_id is not allowed: {update.update_id}"
            )
        return UpdateAcceptanceDecision(
            action=UpdateAcceptanceAction.ACCEPT,
            update_envelope=_normalize_update(update, accepted_at=accepted_at),
        )


@dataclass(slots=True)
class IdempotentRoundUpdateAcceptancePolicy:
    """동일한 update 재전송은 idempotent accept로 처리하는 정책."""

    def evaluate(
        self,
        *,
        record: RoundRecord,
        update: TrainingUpdateEnvelope,
        accepted_at: datetime,
    ) -> UpdateAcceptanceDecision:
        _validate_update_context(record=record, update=update)
        normalized_update = _normalize_update(update, accepted_at=accepted_at)
        existing_update = _find_existing_update(
            record=record,
            update_id=normalized_update.update_id,
        )
        if existing_update is None:
            return UpdateAcceptanceDecision(
                action=UpdateAcceptanceAction.ACCEPT,
                update_envelope=normalized_update,
            )
        if _idempotency_fingerprint(existing_update) != _idempotency_fingerprint(
            normalized_update
        ):
            raise RoundConflictError(
                "Conflicting duplicate update_id is not allowed: "
                f"{normalized_update.update_id}"
            )
        return UpdateAcceptanceDecision(
            action=UpdateAcceptanceAction.IDEMPOTENT,
            update_envelope=existing_update,
        )

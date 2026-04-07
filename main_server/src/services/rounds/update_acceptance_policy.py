"""Round update acceptance/idempotency 정책."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from main_server.src.services.rounds.models import RoundRecord
from shared.src.contracts.training_contracts import TrainingUpdateEnvelope


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


class RoundNetworkPolicy(Protocol):
    """중복/재전송 같은 네트워크 수명주기 정책."""

    def evaluate(
        self,
        *,
        record: RoundRecord,
        update: TrainingUpdateEnvelope,
        accepted_at: datetime,
    ) -> UpdateAcceptanceDecision:
        """update를 accept할지, idempotent로 처리할지 결정한다."""


class RoundTrustPolicy(Protocol):
    """데이터 신뢰도와 제출 주체 규칙을 판단하는 정책."""

    def evaluate(
        self,
        *,
        record: RoundRecord,
        update: TrainingUpdateEnvelope,
        accepted_at: datetime,
    ) -> TrainingUpdateEnvelope:
        """update를 허용하거나 거부하고, 필요시 보정된 update를 돌려준다."""


def _normalize_update(
    update: TrainingUpdateEnvelope,
    *,
    accepted_at: datetime,
) -> TrainingUpdateEnvelope:
    if update.created_at is not None:
        return update
    return update.model_copy(update={"created_at": accepted_at})


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


def _find_existing_agent_update(
    *,
    record: RoundRecord,
    agent_id: str,
) -> TrainingUpdateEnvelope | None:
    for existing in record.updates:
        if existing.agent_id == agent_id:
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
class AllowAllRoundTrustPolicy:
    """신뢰도 관점에서 추가 필터링을 하지 않는 기본 정책."""

    def evaluate(
        self,
        *,
        record: RoundRecord,
        update: TrainingUpdateEnvelope,
        accepted_at: datetime,
    ) -> TrainingUpdateEnvelope:
        del record, accepted_at
        return update


@dataclass(slots=True)
class SingleSubmissionPerAgentTrustPolicy:
    """한 round에서 같은 agent_id의 중복 제출을 막는 trust 정책."""

    allow_anonymous: bool = True

    def evaluate(
        self,
        *,
        record: RoundRecord,
        update: TrainingUpdateEnvelope,
        accepted_at: datetime,
    ) -> TrainingUpdateEnvelope:
        del accepted_at
        if update.agent_id is None:
            if self.allow_anonymous:
                return update
            raise RoundValidationError(
                "agent_id is required by the configured trust policy."
            )
        existing = _find_existing_agent_update(record=record, agent_id=update.agent_id)
        if existing is None:
            return update
        raise RoundConflictError(
            f"Duplicate agent_id is not allowed within the round: {update.agent_id}"
        )


@dataclass(slots=True)
class StrictRoundNetworkPolicy:
    """같은 update_id 재전송을 거부하는 네트워크 정책."""

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
class IdempotentRoundNetworkPolicy:
    """동일한 update 재전송은 idempotent accept로 처리하는 네트워크 정책."""

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


@dataclass(slots=True)
class CompositeRoundUpdateAcceptancePolicy:
    """네트워크 정책과 trust 정책을 조합하는 acceptance policy."""

    network_policy: RoundNetworkPolicy = field(default_factory=StrictRoundNetworkPolicy)
    trust_policy: RoundTrustPolicy = field(default_factory=AllowAllRoundTrustPolicy)

    def evaluate(
        self,
        *,
        record: RoundRecord,
        update: TrainingUpdateEnvelope,
        accepted_at: datetime,
    ) -> UpdateAcceptanceDecision:
        _validate_update_context(record=record, update=update)
        trusted_update = self.trust_policy.evaluate(
            record=record,
            update=update,
            accepted_at=accepted_at,
        )
        return self.network_policy.evaluate(
            record=record,
            update=trusted_update,
            accepted_at=accepted_at,
        )


@dataclass(slots=True)
class StrictRoundUpdateAcceptancePolicy:
    """기본 acceptance policy.

    네트워크 수명주기 정책과 데이터 trust 정책을 조합한다.
    """

    network_policy: RoundNetworkPolicy = field(default_factory=StrictRoundNetworkPolicy)
    trust_policy: RoundTrustPolicy = field(default_factory=AllowAllRoundTrustPolicy)

    def evaluate(
        self,
        *,
        record: RoundRecord,
        update: TrainingUpdateEnvelope,
        accepted_at: datetime,
    ) -> UpdateAcceptanceDecision:
        return CompositeRoundUpdateAcceptancePolicy(
            network_policy=self.network_policy,
            trust_policy=self.trust_policy,
        ).evaluate(
            record=record,
            update=update,
            accepted_at=accepted_at,
        )


@dataclass(slots=True)
class IdempotentRoundUpdateAcceptancePolicy:
    """idempotent 네트워크 정책을 쓰는 acceptance policy."""

    network_policy: RoundNetworkPolicy = field(
        default_factory=IdempotentRoundNetworkPolicy
    )
    trust_policy: RoundTrustPolicy = field(default_factory=AllowAllRoundTrustPolicy)

    def evaluate(
        self,
        *,
        record: RoundRecord,
        update: TrainingUpdateEnvelope,
        accepted_at: datetime,
    ) -> UpdateAcceptanceDecision:
        return CompositeRoundUpdateAcceptancePolicy(
            network_policy=self.network_policy,
            trust_policy=self.trust_policy,
        ).evaluate(
            record=record,
            update=update,
            accepted_at=accepted_at,
        )

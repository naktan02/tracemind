"""Round trust policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from main_server.src.services.federation.rounds.acceptance.errors import (
    RoundConflictError,
    RoundValidationError,
)
from main_server.src.services.federation.rounds.acceptance.helpers import (
    find_existing_agent_update,
)
from main_server.src.services.federation.rounds.models import RoundRecord
from shared.src.contracts.training_contracts import TrainingUpdateEnvelope


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
        existing = find_existing_agent_update(record=record, agent_id=update.agent_id)
        if existing is None:
            return update
        raise RoundConflictError(
            f"Duplicate agent_id is not allowed within the round: {update.agent_id}"
        )

"""Round network lifecycle policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from main_server.src.services.federation.rounds.acceptance.errors import (
    RoundConflictError,
)
from main_server.src.services.federation.rounds.acceptance.helpers import (
    build_idempotency_fingerprint,
    find_existing_update,
    normalize_update,
    validate_update_context,
)
from main_server.src.services.federation.rounds.acceptance.models import (
    UpdateAcceptanceAction,
    UpdateAcceptanceDecision,
)
from main_server.src.services.federation.rounds.models import RoundRecord
from shared.src.contracts.training_contracts import TrainingUpdateEnvelope


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
        validate_update_context(record=record, update=update)
        if find_existing_update(record=record, update_id=update.update_id) is not None:
            raise RoundConflictError(
                f"Duplicate update_id is not allowed: {update.update_id}"
            )
        return UpdateAcceptanceDecision(
            action=UpdateAcceptanceAction.ACCEPT,
            update_envelope=normalize_update(update, accepted_at=accepted_at),
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
        validate_update_context(record=record, update=update)
        normalized_update = normalize_update(update, accepted_at=accepted_at)
        existing_update = find_existing_update(
            record=record,
            update_id=normalized_update.update_id,
        )
        if existing_update is None:
            return UpdateAcceptanceDecision(
                action=UpdateAcceptanceAction.ACCEPT,
                update_envelope=normalized_update,
            )
        if build_idempotency_fingerprint(
            existing_update
        ) != build_idempotency_fingerprint(normalized_update):
            raise RoundConflictError(
                "Conflicting duplicate update_id is not allowed: "
                f"{normalized_update.update_id}"
            )
        return UpdateAcceptanceDecision(
            action=UpdateAcceptanceAction.IDEMPOTENT,
            update_envelope=existing_update,
        )

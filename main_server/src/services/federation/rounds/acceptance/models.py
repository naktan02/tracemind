"""Round acceptance contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from main_server.src.services.federation.rounds.boundary.models import RoundRecord
from shared.src.contracts.training_contracts import TrainingUpdateEnvelope


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

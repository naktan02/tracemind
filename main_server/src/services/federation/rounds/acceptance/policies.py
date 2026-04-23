"""Composite round acceptance policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from main_server.src.services.federation.rounds.acceptance.helpers import (
    validate_update_context,
)
from main_server.src.services.federation.rounds.acceptance.models import (
    RoundNetworkPolicy,
    RoundTrustPolicy,
    UpdateAcceptanceDecision,
)
from main_server.src.services.federation.rounds.acceptance.network_policies import (
    IdempotentRoundNetworkPolicy,
    StrictRoundNetworkPolicy,
)
from main_server.src.services.federation.rounds.acceptance.trust_policies import (
    AllowAllRoundTrustPolicy,
)
from main_server.src.services.federation.rounds.models import RoundRecord
from shared.src.contracts.training_contracts import TrainingUpdateEnvelope


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
        validate_update_context(record=record, update=update)
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
    """기본 acceptance policy."""

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

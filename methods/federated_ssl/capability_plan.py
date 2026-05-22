"""FL SSL runtime capability plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from methods.federated.aggregation_weighting import AggregationWeightPolicy
from methods.federated.client_split import (
    LABELED_EXPOSURE_POLICY_NAMES,
    LABELED_EXPOSURE_SHARED_CLIENT_SEED,
)
from methods.federated.participation import ClientParticipationPolicy

LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED = "client_labeled_and_unlabeled"
LOCAL_SUPERVISION_CLIENT_UNLABELED_ONLY = "client_unlabeled_only"
LOCAL_SUPERVISION_SERVER_LABELED_ONLY = "server_labeled_only"
LOCAL_SUPERVISION_REGIMES = frozenset(
    {
        LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED,
        LOCAL_SUPERVISION_CLIENT_UNLABELED_ONLY,
        LOCAL_SUPERVISION_SERVER_LABELED_ONLY,
    }
)

SERVER_STEP_NONE = "none"
SERVER_STEP_SUPERVISED_SEED = "supervised_seed_step"
SERVER_STEP_POLICIES = frozenset({SERVER_STEP_NONE, SERVER_STEP_SUPERVISED_SEED})

PEER_CONTEXT_NONE = "none"
PEER_CONTEXT_PREDICTION_SIMILARITY_TOPK = "prediction_similarity_topk"
PEER_CONTEXT_POLICIES = frozenset(
    {
        PEER_CONTEXT_NONE,
        PEER_CONTEXT_PREDICTION_SIMILARITY_TOPK,
    }
)

UPDATE_PARTITION_UNIFIED = "unified"
UPDATE_PARTITION_PARTITIONED = "partitioned"
UPDATE_PARTITION_POLICIES = frozenset(
    {
        UPDATE_PARTITION_UNIFIED,
        UPDATE_PARTITION_PARTITIONED,
    }
)

QUERY_MULTIVIEW_SOURCE_MATERIALIZED_ROWS = "materialized_rows"
QUERY_MULTIVIEW_SOURCE_AGENT_GENERATED_OR_CACHED = "agent_generated_or_cached"
QUERY_MULTIVIEW_SOURCES = frozenset(
    {
        QUERY_MULTIVIEW_SOURCE_MATERIALIZED_ROWS,
        QUERY_MULTIVIEW_SOURCE_AGENT_GENERATED_OR_CACHED,
    }
)


@dataclass(frozen=True, slots=True)
class FederatedSslCapabilityPlan:
    """simulation/live runtime이 공통으로 해석하는 FL SSL capability 조합."""

    client_participation_policy: ClientParticipationPolicy
    aggregation_weight_policy: AggregationWeightPolicy
    labeled_exposure_policy_name: str = LABELED_EXPOSURE_SHARED_CLIENT_SEED
    local_supervision_regime_name: str = LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED
    server_step_policy_name: str = SERVER_STEP_NONE
    peer_context_policy_name: str = PEER_CONTEXT_NONE
    update_partition_policy_name: str = UPDATE_PARTITION_UNIFIED
    query_multiview_source_name: str = QUERY_MULTIVIEW_SOURCE_MATERIALIZED_ROWS

    @classmethod
    def from_mappings(
        cls,
        *,
        client_participation_policy: Mapping[str, object] | None,
        aggregation_weight_policy: Mapping[str, object] | None,
        labeled_exposure_policy: Mapping[str, object] | None,
        local_supervision_regime: Mapping[str, object] | None,
        server_step_policy: Mapping[str, object] | None,
        peer_context_policy: Mapping[str, object] | None,
        update_partition_policy: Mapping[str, object] | None,
        query_multiview_source: Mapping[str, object] | None,
    ) -> "FederatedSslCapabilityPlan":
        return cls(
            client_participation_policy=ClientParticipationPolicy.from_mapping(
                _mapping_or_none(client_participation_policy)
            ),
            aggregation_weight_policy=AggregationWeightPolicy.from_mapping(
                aggregation_weight_policy
            ),
            labeled_exposure_policy_name=_policy_name(
                labeled_exposure_policy,
                default=LABELED_EXPOSURE_SHARED_CLIENT_SEED,
            ),
            local_supervision_regime_name=_policy_name(
                local_supervision_regime,
                default=LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED,
            ),
            server_step_policy_name=_policy_name(
                server_step_policy,
                default=SERVER_STEP_NONE,
            ),
            peer_context_policy_name=_policy_name(
                peer_context_policy,
                default=PEER_CONTEXT_NONE,
            ),
            update_partition_policy_name=_policy_name(
                update_partition_policy,
                default=UPDATE_PARTITION_UNIFIED,
            ),
            query_multiview_source_name=_policy_name(
                query_multiview_source,
                default=QUERY_MULTIVIEW_SOURCE_MATERIALIZED_ROWS,
            ),
        )

    def __post_init__(self) -> None:
        _validate_name(
            self.labeled_exposure_policy_name,
            allowed=LABELED_EXPOSURE_POLICY_NAMES,
            field_name="labeled_exposure_policy.name",
        )
        _validate_name(
            self.local_supervision_regime_name,
            allowed=LOCAL_SUPERVISION_REGIMES,
            field_name="local_supervision_regime.name",
        )
        _validate_name(
            self.server_step_policy_name,
            allowed=SERVER_STEP_POLICIES,
            field_name="server_step_policy.name",
        )
        _validate_name(
            self.peer_context_policy_name,
            allowed=PEER_CONTEXT_POLICIES,
            field_name="peer_context_policy.name",
        )
        _validate_name(
            self.update_partition_policy_name,
            allowed=UPDATE_PARTITION_POLICIES,
            field_name="update_partition_policy.name",
        )
        _validate_name(
            self.query_multiview_source_name,
            allowed=QUERY_MULTIVIEW_SOURCES,
            field_name="query_multiview_source.name",
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "client_participation_policy": (
                self.client_participation_policy.to_payload()
            ),
            "labeled_exposure_policy": {
                "name": self.labeled_exposure_policy_name,
            },
            "local_supervision_regime": {
                "name": self.local_supervision_regime_name,
            },
            "server_step_policy": {
                "name": self.server_step_policy_name,
            },
            "peer_context_policy": {
                "name": self.peer_context_policy_name,
            },
            "update_partition_policy": {
                "name": self.update_partition_policy_name,
            },
            "aggregation_weight_policy": self.aggregation_weight_policy.to_payload(),
            "query_multiview_source": {
                "name": self.query_multiview_source_name,
            },
        }


def _mapping_or_none(source: Mapping[str, object] | None) -> dict[str, object] | None:
    if source is None:
        return None
    return dict(source)


def _policy_name(source: Mapping[str, object] | None, *, default: str) -> str:
    if source is None:
        return default
    return str(source.get("name", default)).strip() or default


def _validate_name(
    value: str,
    *,
    allowed: frozenset[str],
    field_name: str,
) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}.")

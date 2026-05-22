"""FL SSL capability axis names and small payload normalizers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

LOCAL_SSL_POLICY_PROFILE_PSEUDO_LABEL = "profile_pseudo_label"
LOCAL_SSL_POLICY_PSEUDOLABEL = "pseudolabel"
LOCAL_SSL_POLICY_FIXMATCH = "fixmatch"
LOCAL_SSL_POLICY_FLEXMATCH = "flexmatch"
LOCAL_SSL_POLICY_FREEMATCH = "freematch"
LOCAL_SSL_POLICY_ADAMATCH = "adamatch"
LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT = "fedmatch_agreement"
LOCAL_SSL_POLICIES_REQUIRING_STATE_SURFACE = frozenset(
    {
        LOCAL_SSL_POLICY_FLEXMATCH,
        LOCAL_SSL_POLICY_FREEMATCH,
        LOCAL_SSL_POLICY_ADAMATCH,
    }
)
LOCAL_SSL_POLICIES_FROM_QUERY_SSL = frozenset(
    {
        LOCAL_SSL_POLICY_PSEUDOLABEL,
        LOCAL_SSL_POLICY_FIXMATCH,
        LOCAL_SSL_POLICY_FLEXMATCH,
        LOCAL_SSL_POLICY_FREEMATCH,
        LOCAL_SSL_POLICY_ADAMATCH,
    }
)
LOCAL_SSL_POLICY_NAMES = frozenset(
    {
        LOCAL_SSL_POLICY_PROFILE_PSEUDO_LABEL,
        *LOCAL_SSL_POLICIES_FROM_QUERY_SSL,
        LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
    }
)

SERVER_UPDATE_FEDAVG_MERGED_DELTA = "fedavg_merged_delta"
SERVER_UPDATE_FEDMATCH_PARTITIONED = "fedmatch_partitioned"
SERVER_UPDATE_POLICY_NAMES = frozenset(
    {
        SERVER_UPDATE_FEDAVG_MERGED_DELTA,
        SERVER_UPDATE_FEDMATCH_PARTITIONED,
    }
)


@dataclass(frozen=True, slots=True)
class LocalSslPolicy:
    """client local SSL objective 선택을 canonical 이름으로 정규화한다."""

    name: str = LOCAL_SSL_POLICY_PROFILE_PSEUDO_LABEL

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object] | None,
    ) -> "LocalSslPolicy":
        if source is None:
            return cls()
        return cls(name=str(source.get("name", LOCAL_SSL_POLICY_PROFILE_PSEUDO_LABEL)))

    def __post_init__(self) -> None:
        normalized = self.name.strip().lower().replace("-", "_")
        object.__setattr__(self, "name", normalized)
        if normalized not in LOCAL_SSL_POLICY_NAMES:
            raise ValueError(
                "local_ssl_policy.name must be one of "
                f"{sorted(LOCAL_SSL_POLICY_NAMES)}."
            )

    @property
    def uses_query_ssl_algorithm(self) -> bool:
        """기존 Query SSL algorithm parameter surface를 재사용하는 정책인지 반환한다."""

        return self.name in LOCAL_SSL_POLICIES_FROM_QUERY_SSL

    @property
    def requires_state_surface(self) -> bool:
        """round/client state 저장 surface가 먼저 필요한 정책인지 반환한다."""

        return self.name in LOCAL_SSL_POLICIES_REQUIRING_STATE_SURFACE

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "parameter_source": self._parameter_source(),
            "requires_state_surface": self.requires_state_surface,
        }

    def _parameter_source(self) -> str:
        if self.uses_query_ssl_algorithm:
            return "query_ssl_method"
        if self.name == LOCAL_SSL_POLICY_PROFILE_PSEUDO_LABEL:
            return "local_update_profile"
        return "method_descriptor"


@dataclass(frozen=True, slots=True)
class ServerUpdatePolicy:
    """server가 client update payload를 어떤 의미로 해석할지 나타낸다."""

    name: str = SERVER_UPDATE_FEDAVG_MERGED_DELTA

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object] | None,
    ) -> "ServerUpdatePolicy":
        if source is None:
            return cls()
        return cls(name=str(source.get("name", SERVER_UPDATE_FEDAVG_MERGED_DELTA)))

    def __post_init__(self) -> None:
        normalized = self.name.strip().lower().replace("-", "_")
        object.__setattr__(self, "name", normalized)
        if normalized not in SERVER_UPDATE_POLICY_NAMES:
            raise ValueError(
                "server_update_policy.name must be one of "
                f"{sorted(SERVER_UPDATE_POLICY_NAMES)}."
            )

    def to_payload(self) -> dict[str, object]:
        return {"name": self.name}

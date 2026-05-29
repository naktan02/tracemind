"""Method-owned update partition policy specs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FederatedSslUpdatePartitionPolicy:
    """local loss routing과 aggregation 대상 partition metadata."""

    policy_name: str
    partition_names: tuple[str, ...] = ()
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.policy_name.strip():
            raise ValueError("policy_name must not be empty.")
        normalized = tuple(
            str(name).strip() for name in self.partition_names if str(name).strip()
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("partition_names must be unique.")
        object.__setattr__(self, "partition_names", normalized)

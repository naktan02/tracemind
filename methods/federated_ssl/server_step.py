"""Method-owned FL SSL server step policy specs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FederatedSslServerStepPolicy:
    """server-side update policy metadata."""

    policy_name: str
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.policy_name.strip():
            raise ValueError("policy_name must not be empty.")

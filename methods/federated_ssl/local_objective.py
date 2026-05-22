"""Method-owned FL SSL local objective port."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class FederatedSslLocalObjectiveSpec:
    """client local objective의 method-owned metadata."""

    objective_name: str
    required_batch_views: tuple[str, ...] = ()
    metric_prefix: str = "local_objective"
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objective_name.strip():
            raise ValueError("objective_name must not be empty.")
        if not self.metric_prefix.strip():
            raise ValueError("metric_prefix must not be empty.")


class FederatedSslLocalObjective(Protocol):
    """runtime adapter가 method-owned local objective를 호출하는 port."""

    spec: FederatedSslLocalObjectiveSpec

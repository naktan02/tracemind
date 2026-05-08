"""Experiment compile policy registry primitive."""

from __future__ import annotations

from dataclasses import dataclass, field

from main_server.src.services.experiment_workspace.compiler.contracts import (
    ExperimentCompilePolicy,
    NoOpExperimentCompilePolicy,
)


@dataclass(slots=True)
class ExperimentCompilePolicyRegistry:
    """Entrypoint 이름별 compile policy registry."""

    _policies: dict[str, ExperimentCompilePolicy] = field(default_factory=dict)
    _default_policy: ExperimentCompilePolicy = field(
        default_factory=NoOpExperimentCompilePolicy
    )

    def register(
        self,
        entrypoint_name: str,
        policy: ExperimentCompilePolicy,
    ) -> None:
        self._policies[entrypoint_name] = policy

    def resolve(self, entrypoint_name: str) -> ExperimentCompilePolicy:
        return self._policies.get(entrypoint_name, self._default_policy)

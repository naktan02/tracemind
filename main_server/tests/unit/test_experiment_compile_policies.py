"""Experiment compile policy registry tests."""

from __future__ import annotations

from dataclasses import dataclass

from main_server.src.services.experiment_workspace.compiler.policies import (
    ExperimentCompileContext,
    ExperimentCompilePolicyRegistry,
    NoOpExperimentCompilePolicy,
)


@dataclass(frozen=True, slots=True)
class _TestCompilePolicy:
    def collect_warnings(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> tuple[str, ...]:
        del context
        return ("test warning",)

    def validate_requirements(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> None:
        del context


def test_experiment_compile_policy_registry_returns_noop_policy_by_default() -> None:
    registry = ExperimentCompilePolicyRegistry()

    policy = registry.resolve("unknown_entrypoint")

    assert isinstance(policy, NoOpExperimentCompilePolicy)


def test_experiment_compile_policy_registry_returns_registered_policy() -> None:
    registry = ExperimentCompilePolicyRegistry()
    policy = _TestCompilePolicy()

    registry.register("train_custom_entrypoint", policy)

    assert registry.resolve("train_custom_entrypoint") is policy

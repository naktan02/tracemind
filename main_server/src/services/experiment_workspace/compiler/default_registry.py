"""Default experiment compile policy wiring."""

from __future__ import annotations

from main_server.src.services.experiment_workspace.compiler.entrypoint_policies import (
    FederatedSimulationCompilePolicy,
    FixMatchCompilePolicy,
)
from main_server.src.services.experiment_workspace.compiler.registry import (
    ExperimentCompilePolicyRegistry,
)


def build_default_experiment_compile_policy_registry() -> (
    ExperimentCompilePolicyRegistry
):
    """Built-in entrypoint compile policies를 registry에 연결한다."""

    registry = ExperimentCompilePolicyRegistry()
    registry.register(
        "run_federated_simulation",
        FederatedSimulationCompilePolicy(),
    )
    registry.register(
        "train_lora_fixmatch",
        FixMatchCompilePolicy(),
    )
    return registry


DEFAULT_EXPERIMENT_COMPILE_POLICY_REGISTRY = (
    build_default_experiment_compile_policy_registry()
)

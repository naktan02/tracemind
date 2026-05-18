"""FL simulation runtime/objective compatibility adapter."""

from __future__ import annotations

from methods.adaptation.runtime_objective_compatibility import (
    require_adapter_runtime_matches_objective,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    SimulationRunRequest,
)


def require_round_runtime_matches_training_objective(
    request: SimulationRunRequest,
) -> None:
    """round runtime family state와 local objective payload config drift를 막는다."""

    require_adapter_runtime_matches_objective(
        adapter_kind=request.round_runtime_config.adapter_family_name,
        runtime_config=request.round_runtime_config.runtime_payload_for_adapter_family(),
        objective_config=request.training_task_config.objective_config,
    )

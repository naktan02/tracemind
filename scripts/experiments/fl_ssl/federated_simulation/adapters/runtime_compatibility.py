"""FL simulation runtime/objective compatibility adapter."""

from __future__ import annotations

from methods.adaptation.lora_classifier.runtime_compatibility import (
    require_lora_classifier_runtime_matches_objective,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    SimulationRunRequest,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
)


def require_round_runtime_matches_training_objective(
    request: SimulationRunRequest,
) -> None:
    """round runtime family state와 local objective payload config drift를 막는다."""

    adapter_family_name = (
        request.round_runtime_config.adapter_family_name.strip().lower()
    )
    if adapter_family_name != LORA_CLASSIFIER_ADAPTER_KIND:
        return
    lora_runtime_config = request.round_runtime_config.lora_classifier
    if lora_runtime_config is None:
        raise ValueError(
            "lora_classifier round runtime requires lora_classifier bootstrap config."
        )
    require_lora_classifier_runtime_matches_objective(
        runtime_config=lora_runtime_config,
        objective_config=request.training_task_config.objective_config,
    )

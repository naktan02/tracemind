"""Federated simulation용 classifier validation adapter."""

from __future__ import annotations

from methods.adaptation.lora_classifier.evaluation import (
    LORA_CLASSIFIER_EVALUATOR_NAME,
    evaluate_lora_classifier_validation_payload,
    require_lora_classifier_state,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    SimulationEvaluation,
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_server.lora_classifier_state import (
    materialize_simulation_lora_classifier_base_state,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


def evaluate_simulation_validation(
    *,
    request: SimulationRunRequest,
    active: ActiveSimulationState,
    rows: list[LabeledQueryRow],
    objective_config: TrainingObjectiveConfig | None,
) -> SimulationEvaluation:
    """LoRA-classifier state 기준 validation row를 평가한다."""

    if request.validation_config.scorer_backend_name != LORA_CLASSIFIER_EVALUATOR_NAME:
        raise ValueError(
            "FL SSL simulation validation only supports lora_classifier_eval. "
            "Prototype-based validation was removed from this simulation path."
        )
    if not rows:
        return SimulationEvaluation(row_count=0, top1_accuracy=0.0, accepted_ratio=0.0)
    adapter_state = require_lora_classifier_state(active.adapter_state)
    payload = evaluate_lora_classifier_validation_payload(
        rows=rows,
        adapter_state=adapter_state,
        base_parameters=materialize_simulation_lora_classifier_base_state(
            output_dir=request.output_dir,
            adapter_state=adapter_state,
        ),
        objective_config=objective_config,
        runtime_config=request.local_trainer_runtime_config,
        batch_size=request.training_task_config.batch_size,
        seed=request.seed,
    )
    return SimulationEvaluation(**payload)

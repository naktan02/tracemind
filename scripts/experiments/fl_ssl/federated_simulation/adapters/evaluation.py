"""Federated simulation validation adapter."""

from __future__ import annotations

from typing import Any

from methods.common.runtime_resources import RuntimeResourceCache
from scripts.configured_callable import load_configured_callable
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    SimulationEvaluation,
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_server.aggregation_artifacts import (
    build_simulation_aggregation_context,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


def evaluate_simulation_validation(
    *,
    request: SimulationRunRequest,
    active: ActiveSimulationState,
    rows: list[LabeledQueryRow],
    objective_config: TrainingObjectiveConfig | None,
    runtime_resource_cache: RuntimeResourceCache | None = None,
) -> SimulationEvaluation:
    """config-declared validation evaluator로 validation row를 평가한다."""

    if not rows:
        return SimulationEvaluation(row_count=0, top1_accuracy=0.0, accepted_ratio=0.0)
    evaluator = _load_validation_evaluator(
        _validation_evaluator_path(request.round_runtime_config)
    )
    payload = evaluator(
        rows=rows,
        adapter_state=active.adapter_state,
        aggregation_context=build_simulation_aggregation_context(
            output_dir=request.output_dir,
            next_model_revision=active.adapter_state.model_revision,
            aggregated_at=active.adapter_state.updated_at,
        ),
        objective_config=objective_config,
        runtime_config=request.local_trainer_runtime_config,
        batch_size=request.training_task_config.batch_size,
        seed=request.seed,
        scorer_backend_name=request.validation_config.scorer_backend_name,
        runtime_resource_cache=runtime_resource_cache,
    )
    return SimulationEvaluation(**payload)


def _validation_evaluator_path(round_runtime_config: object) -> str:
    raw_value = getattr(round_runtime_config, "validation_evaluator", None)
    evaluator_path = "" if raw_value is None else str(raw_value).strip()
    if not evaluator_path:
        raise ValueError("round_runtime.validation_evaluator is required.")
    return evaluator_path


def _load_validation_evaluator(evaluator_path: str) -> Any:
    return load_configured_callable(
        evaluator_path,
        field_name="round_runtime.validation_evaluator",
    )

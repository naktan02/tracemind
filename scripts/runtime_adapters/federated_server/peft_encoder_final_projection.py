"""PEFT encoder final projection runtime adapter."""

from __future__ import annotations

from typing import Any

from methods.adaptation.peft_text_encoder.evaluation import (
    require_peft_encoder_state,
)
from methods.adaptation.peft_text_encoder.simulation_runtime.final_projection import (
    build_peft_encoder_final_projection_artifacts_from_state,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_server.aggregation_artifacts import (
    build_simulation_aggregation_context,
)


def build_peft_encoder_final_projection_artifacts(
    *,
    request: SimulationRunRequest,
    active: ActiveSimulationState,
    runtime_resource_cache: Any | None = None,
) -> dict[str, Any] | None:
    """최종 global PEFT encoder state projection artifact를 만든다."""

    adapter_state = require_peft_encoder_state(active.adapter_state)
    return build_peft_encoder_final_projection_artifacts_from_state(
        rows_by_dataset_name=_projection_rows_by_dataset_name(request),
        adapter_state=adapter_state,
        aggregation_context=build_simulation_aggregation_context(
            output_dir=request.output_dir,
            next_model_revision=adapter_state.model_revision,
            aggregated_at=adapter_state.updated_at,
        ),
        objective_config=request.training_task_config.objective_config,
        runtime_config=request.local_trainer_runtime_config,
        batch_size=int(request.training_task_config.batch_size),
        projection_dir=request.output_dir / "projections",
        seed=request.seed,
        runtime_resource_cache=runtime_resource_cache,
    )


def _projection_rows_by_dataset_name(
    request: SimulationRunRequest,
) -> dict[str, Any]:
    rows_by_name = {
        "validation": request.validation_rows,
        "test": request.test_rows,
    }
    return {
        dataset_name: rows
        for dataset_name, rows in rows_by_name.items()
        if dataset_name in request.final_projection_config.dataset_names and rows
    }

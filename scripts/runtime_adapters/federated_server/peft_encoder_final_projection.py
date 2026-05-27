"""PEFT encoder final projection runtime adapter."""

from __future__ import annotations

from typing import Any

from methods.adaptation.query_text_views.data import build_dataloader
from methods.adaptation.text_classifier.peft_encoder.evaluation import (
    require_peft_encoder_state,
)
from methods.adaptation.text_classifier.peft_encoder.projection_artifacts import (
    write_peft_encoder_projection_artifacts,
)
from methods.adaptation.text_classifier.peft_encoder.runtime_family import (
    build_training_backend_config_for_peft_encoder_state,
)
from methods.adaptation.text_classifier.peft_encoder.training.delta_extraction import (
    load_peft_encoder_base_parameters_into_model,
)
from methods.adaptation.text_classifier.peft_encoder.training.loops import set_seed
from methods.adaptation.text_classifier.peft_encoder.training.modeling import (
    build_peft_encoder_text_classifier_from_config,
)
from methods.adaptation.text_classifier.peft_encoder.update.materialization import (
    materialize_base_peft_encoder_state,
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
    """최종 global PEFT-backed classifier state projection artifact를 만든다."""

    adapter_state = require_peft_encoder_state(active.adapter_state)
    labels = [str(label) for label in adapter_state.label_schema]
    lora_config = build_training_backend_config_for_peft_encoder_state(
        active_adapter_state=adapter_state,
        objective_config=request.training_task_config.objective_config,
    )
    set_seed(request.seed)
    model, tokenizer = build_peft_encoder_text_classifier_from_config(
        labels=labels,
        lora_config=lora_config,
        runtime_config=request.local_trainer_runtime_config,
        runtime_resource_cache=runtime_resource_cache,
    )
    load_peft_encoder_base_parameters_into_model(
        model=model,
        labels=labels,
        base_parameters=materialize_base_peft_encoder_state(
            base_state=adapter_state,
            context=build_simulation_aggregation_context(
                output_dir=request.output_dir,
                next_model_revision=adapter_state.model_revision,
                aggregated_at=adapter_state.updated_at,
            ),
        ),
        device=request.local_trainer_runtime_config.device,
    )
    eval_loaders = _build_projection_eval_loaders(
        request=request,
        tokenizer=tokenizer,
        labels=labels,
        max_length=int(lora_config.max_length),
        task_prefix=lora_config.task_prefix,
    )
    if not eval_loaders:
        return {"enabled": False, "reason": "no_projection_datasets"}
    return write_peft_encoder_projection_artifacts(
        model=model,
        eval_loaders=eval_loaders,
        categories=labels,
        device=request.local_trainer_runtime_config.device,
        projection_dir=request.output_dir / "projections",
        seed=request.seed,
        schema_version="fl_ssl_final_projection_artifacts.v1",
    ) or {"enabled": False, "reason": "projection_writer_returned_none"}


def _build_projection_eval_loaders(
    *,
    request: SimulationRunRequest,
    tokenizer: Any,
    labels: list[str],
    max_length: int,
    task_prefix: str,
) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    rows_by_name = {
        "validation": request.validation_rows,
        "test": request.test_rows,
    }
    return {
        dataset_name: build_dataloader(
            rows=list(rows),
            label_to_index=label_to_index,
            tokenizer=tokenizer,
            batch_size=int(request.training_task_config.batch_size),
            max_length=max_length,
            task_prefix=task_prefix,
            shuffle=False,
        )
        for dataset_name, rows in rows_by_name.items()
        if dataset_name in request.final_projection_config.dataset_names and rows
    }

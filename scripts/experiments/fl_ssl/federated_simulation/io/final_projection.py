"""FL SSL final global model projection artifact builder."""

from __future__ import annotations

from typing import Any

from methods.adaptation.lora_classifier.config import (
    build_lora_classifier_training_backend_config,
)
from methods.adaptation.lora_classifier.evaluation import require_lora_classifier_state
from methods.adaptation.lora_classifier.training.delta_extraction import (
    load_lora_classifier_base_parameters_into_model,
)
from methods.adaptation.lora_classifier.training.loops import set_seed
from methods.adaptation.lora_classifier.training.modeling import (
    build_lora_text_classifier_from_config,
)
from methods.adaptation.query_classifier_adaptation.data import build_dataloader
from methods.adaptation.text_classifier.peft_encoder.update.materialization import (
    materialize_base_lora_classifier_state,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    SimulationRunRequest,
)
from scripts.experiments.lora_classifier_projection import (
    write_lora_classifier_projection_artifacts,
)
from scripts.runtime_adapters.federated_server.aggregation_artifacts import (
    build_simulation_aggregation_context,
)


def build_final_projection_artifacts(
    *,
    request: SimulationRunRequest,
    active: ActiveSimulationState,
    runtime_resource_cache: Any | None = None,
) -> dict[str, Any] | None:
    """최종 global LoRA state의 validation/test projection artifact를 만든다."""

    config = request.final_projection_config
    if not config.enabled:
        return {"enabled": False, "reason": "disabled_by_config"}
    try:
        return _build_final_projection_artifacts(
            request=request,
            active=active,
            runtime_resource_cache=runtime_resource_cache,
        )
    except Exception as exc:
        if config.fail_on_error:
            raise
        return {
            "enabled": False,
            "reason": "projection_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _build_final_projection_artifacts(
    *,
    request: SimulationRunRequest,
    active: ActiveSimulationState,
    runtime_resource_cache: Any | None,
) -> dict[str, Any]:
    adapter_state = require_lora_classifier_state(active.adapter_state)
    labels = [str(label) for label in adapter_state.label_schema]
    lora_config = build_lora_classifier_training_backend_config(
        request.training_task_config.objective_config
    )
    set_seed(request.seed)
    model, tokenizer = build_lora_text_classifier_from_config(
        labels=labels,
        lora_config=lora_config,
        runtime_config=request.local_trainer_runtime_config,
        runtime_resource_cache=runtime_resource_cache,
    )
    load_lora_classifier_base_parameters_into_model(
        model=model,
        labels=labels,
        base_parameters=materialize_base_lora_classifier_state(
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
    return write_lora_classifier_projection_artifacts(
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

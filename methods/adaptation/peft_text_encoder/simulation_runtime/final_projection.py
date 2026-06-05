"""PEFT text encoder simulation final projection core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from methods.adaptation.peft_text_encoder.evaluation import (
    require_peft_encoder_state,
)
from methods.adaptation.peft_text_encoder.projection_artifacts import (
    write_peft_encoder_projection_artifacts,
)
from methods.adaptation.peft_text_encoder.training.delta_extraction import (
    load_peft_encoder_base_parameters_into_model,
)
from methods.adaptation.peft_text_encoder.training.loops import set_seed
from methods.adaptation.peft_text_encoder.training.modeling import (
    PeftEncoderModelRuntimeConfig,
    build_peft_text_encoder_with_linear_head_from_config,
)
from methods.adaptation.peft_text_encoder.training.pseudo_label_diagnostics import (
    tokenization_cache_namespace,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    materialize_base_peft_encoder_state,
)
from methods.adaptation.peft_text_encoder.update_family_runtime import (
    build_training_backend_config_for_peft_encoder_state,
)
from methods.adaptation.query_text_views.data import build_dataloader
from methods.adaptation.query_text_views.tokenization import (
    resolve_text_tokenization_cache,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.federated.aggregation.base import FederatedAggregationContext
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


def build_peft_encoder_final_projection_artifacts_from_state(
    *,
    rows_by_dataset_name: Mapping[str, Sequence[LabeledQueryRow]],
    adapter_state: object,
    aggregation_context: FederatedAggregationContext,
    objective_config: TrainingObjectiveConfig | None,
    runtime_config: PeftEncoderModelRuntimeConfig,
    batch_size: int,
    projection_dir: Path,
    seed: int,
    runtime_resource_cache: RuntimeResourceCache | None = None,
) -> dict[str, Any] | None:
    """최종 global PEFT encoder state의 projection artifact를 만든다."""

    state = require_peft_encoder_state(adapter_state)
    labels = [str(label) for label in state.label_schema]
    peft_config = build_training_backend_config_for_peft_encoder_state(
        active_adapter_state=state,
        objective_config=objective_config,
    )
    set_seed(seed)
    model, tokenizer = build_peft_text_encoder_with_linear_head_from_config(
        labels=labels,
        peft_config=peft_config,
        runtime_config=runtime_config,
        runtime_resource_cache=runtime_resource_cache,
    )
    load_peft_encoder_base_parameters_into_model(
        model=model,
        labels=labels,
        base_parameters=materialize_base_peft_encoder_state(
            base_state=state,
            context=aggregation_context,
        ),
        device=runtime_config.device,
    )
    eval_loaders = _build_projection_eval_loaders(
        rows_by_dataset_name=rows_by_dataset_name,
        tokenizer=tokenizer,
        labels=labels,
        batch_size=batch_size,
        max_length=int(peft_config.max_length),
        task_prefix=peft_config.task_prefix,
        tokenization_cache=resolve_text_tokenization_cache(runtime_resource_cache),
        tokenization_cache_namespace=tokenization_cache_namespace(peft_config),
    )
    if not eval_loaders:
        return {"enabled": False, "reason": "no_projection_datasets"}
    return write_peft_encoder_projection_artifacts(
        model=model,
        eval_loaders=eval_loaders,
        categories=labels,
        device=runtime_config.device,
        projection_dir=projection_dir,
        seed=seed,
        schema_version="fl_ssl_final_projection_artifacts.v1",
    ) or {"enabled": False, "reason": "projection_writer_returned_none"}


def _build_projection_eval_loaders(
    *,
    rows_by_dataset_name: Mapping[str, Sequence[LabeledQueryRow]],
    tokenizer: Any,
    labels: list[str],
    batch_size: int,
    max_length: int,
    task_prefix: str,
    tokenization_cache: Any | None,
    tokenization_cache_namespace: str,
) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    return {
        dataset_name: build_dataloader(
            rows=list(rows),
            label_to_index=label_to_index,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            task_prefix=task_prefix,
            shuffle=False,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace,
        )
        for dataset_name, rows in rows_by_dataset_name.items()
        if rows
    }

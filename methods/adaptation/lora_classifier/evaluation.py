"""LoRA-classifier global state 평가 core."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
)
from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
    build_lora_classifier_training_backend_config,
)
from methods.adaptation.lora_classifier.training.delta_extraction import (
    load_lora_classifier_base_parameters_into_model,
)
from methods.adaptation.lora_classifier.training.loops import (
    evaluate_classifier,
    set_seed,
)
from methods.adaptation.lora_classifier.training.modeling import (
    LoraClassifierModelRuntimeConfig,
    build_lora_text_classifier_from_config,
)
from methods.adaptation.query_classifier_adaptation.data import build_dataloader
from methods.common.runtime_resources import RuntimeResourceCache
from methods.evaluation.classification_payload import (
    build_classification_evaluation_payload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierState,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

LORA_CLASSIFIER_EVALUATOR_NAME = "lora_classifier_eval"
LORA_CLASSIFIER_EVALUATION_DISTRIBUTION_KIND = "lora_classifier_logits_softmax"
LORA_CLASSIFIER_EVALUATION_CONFIDENCE_KIND = "lora_classifier_top1_probability"


def evaluate_lora_classifier_state(
    *,
    rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
    base_parameters: LoraClassifierMaterializedState,
    lora_config: LoraClassifierTrainingBackendConfig,
    runtime_config: LoraClassifierModelRuntimeConfig,
    batch_size: int,
    seed: int,
    runtime_resource_cache: RuntimeResourceCache | None = None,
) -> dict[str, Any]:
    """materialized global LoRA/head state로 labeled rows를 평가한다."""

    effective_labels = [str(label) for label in labels]
    if not effective_labels:
        raise ValueError("LoRA-classifier evaluation labels must not be empty.")
    if batch_size <= 0:
        raise ValueError("LoRA-classifier evaluation batch_size must be positive.")

    set_seed(int(seed))
    model, tokenizer = build_lora_text_classifier_from_config(
        labels=effective_labels,
        lora_config=lora_config,
        runtime_config=runtime_config,
        runtime_resource_cache=runtime_resource_cache,
    )
    load_lora_classifier_base_parameters_into_model(
        model=model,
        labels=effective_labels,
        base_parameters=base_parameters,
        device=runtime_config.device,
    )
    label_to_index = {label: index for index, label in enumerate(effective_labels)}
    dataloader = build_dataloader(
        rows=list(rows),
        label_to_index=label_to_index,
        tokenizer=tokenizer,
        batch_size=int(batch_size),
        max_length=int(lora_config.max_length),
        task_prefix=lora_config.task_prefix,
        shuffle=False,
    )
    return evaluate_classifier(
        model=model,
        dataloader=dataloader,
        categories=effective_labels,
        device=runtime_config.device,
    )


def evaluate_lora_classifier_state_payload(
    *,
    rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
    base_parameters: LoraClassifierMaterializedState,
    lora_config: LoraClassifierTrainingBackendConfig,
    runtime_config: LoraClassifierModelRuntimeConfig,
    batch_size: int,
    seed: int,
    runtime_resource_cache: RuntimeResourceCache | None = None,
) -> dict[str, object]:
    """LoRA-classifier global state 평가 결과를 canonical payload로 반환한다."""

    report = evaluate_lora_classifier_state(
        rows=rows,
        labels=labels,
        base_parameters=base_parameters,
        lora_config=lora_config,
        runtime_config=runtime_config,
        batch_size=batch_size,
        seed=seed,
        runtime_resource_cache=runtime_resource_cache,
    )
    row_count = len(rows)
    return build_classification_evaluation_payload(
        report=report,
        row_count=row_count,
        accepted_ratio=1.0 if row_count > 0 else 0.0,
        loss_kind="cross_entropy_from_lora_classifier_logits",
        score_distribution_kind=LORA_CLASSIFIER_EVALUATION_DISTRIBUTION_KIND,
        selection_confidence_kind=LORA_CLASSIFIER_EVALUATION_CONFIDENCE_KIND,
        mean_selection_confidence=float(report["mean_top_1_probability"]),
        mean_selection_margin=float(report["mean_margin_top1_top2"]),
    )


def evaluate_lora_classifier_validation_payload(
    *,
    rows: Sequence[LabeledQueryRow],
    adapter_state: object,
    base_parameters: LoraClassifierMaterializedState,
    objective_config: TrainingObjectiveConfig | None,
    runtime_config: LoraClassifierModelRuntimeConfig,
    batch_size: int,
    seed: int,
    runtime_resource_cache: RuntimeResourceCache | None = None,
) -> dict[str, object]:
    """FL validation runtime이 넘긴 LoRA state를 method-owned evaluator로 평가한다."""

    if not isinstance(adapter_state, LoraClassifierState):
        raise ValueError(
            f"{LORA_CLASSIFIER_EVALUATOR_NAME!r} requires LoraClassifierState; "
            f"got {type(adapter_state).__name__}."
        )
    return evaluate_lora_classifier_state_payload(
        rows=rows,
        labels=adapter_state.label_schema,
        base_parameters=base_parameters,
        lora_config=build_lora_classifier_training_backend_config(objective_config),
        runtime_config=runtime_config,
        batch_size=batch_size,
        seed=seed,
        runtime_resource_cache=runtime_resource_cache,
    )


def require_lora_classifier_validation_backend(
    *,
    adapter_state: object,
    scorer_backend_name: str,
    prototype_scorer_backend_name: str,
) -> None:
    """LoRA-classifier state에 prototype scorer validation이 붙는 drift를 막는다."""

    if not isinstance(adapter_state, LoraClassifierState):
        return
    if scorer_backend_name == LORA_CLASSIFIER_EVALUATOR_NAME:
        return
    raise ValueError(
        "LoRA-classifier validation must use "
        f"{LORA_CLASSIFIER_EVALUATOR_NAME!r}. "
        f"{prototype_scorer_backend_name!r} is prototype/selection-only "
        "and does not read LoRA/classifier global state."
    )


def require_lora_classifier_state(adapter_state: object) -> LoraClassifierState:
    """runtime adapter가 넘긴 shared state를 LoRA-classifier state로 검증한다."""

    if not isinstance(adapter_state, LoraClassifierState):
        raise ValueError(
            f"{LORA_CLASSIFIER_EVALUATOR_NAME!r} requires LoraClassifierState; "
            f"got {type(adapter_state).__name__}."
        )
    return adapter_state

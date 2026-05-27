"""PEFT-backed classifier global state 평가 core."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from methods.adaptation.peft_text_classifier.config import (
    PeftEncoderTrainingBackendConfig,
    build_lora_classifier_training_backend_config,
    build_peft_classifier_training_backend_config,
)
from methods.adaptation.peft_text_classifier.training.delta_extraction import (
    load_peft_encoder_base_parameters_into_model,
)
from methods.adaptation.peft_text_classifier.training.loops import (
    evaluate_classifier,
    set_seed,
)
from methods.adaptation.peft_text_classifier.training.modeling import (
    LoraClassifierModelRuntimeConfig,
    build_peft_encoder_text_classifier_from_config,
)
from methods.adaptation.peft_text_classifier.update.materialization import (
    PeftEncoderMaterializedState,
    materialize_base_peft_encoder_state,
)
from methods.adaptation.query_text_views.data import build_dataloader
from methods.common.runtime_resources import RuntimeResourceCache
from methods.evaluation.classification_payload import (
    build_classification_evaluation_payload,
)
from methods.federated.aggregation.base import FederatedAggregationContext
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierState,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierState,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

LORA_CLASSIFIER_EVALUATOR_NAME = "lora_classifier_eval"
PEFT_CLASSIFIER_EVALUATOR_NAME = "peft_classifier_eval"
PEFT_CLASSIFIER_ACCEPTED_EVALUATOR_NAMES = (
    PEFT_CLASSIFIER_EVALUATOR_NAME,
    LORA_CLASSIFIER_EVALUATOR_NAME,
)
LORA_CLASSIFIER_EVALUATION_DISTRIBUTION_KIND = "lora_classifier_logits_softmax"
LORA_CLASSIFIER_EVALUATION_CONFIDENCE_KIND = "lora_classifier_top1_probability"
PEFT_CLASSIFIER_EVALUATION_DISTRIBUTION_KIND = "peft_classifier_logits_softmax"
PEFT_CLASSIFIER_EVALUATION_CONFIDENCE_KIND = "peft_classifier_top1_probability"


def evaluate_peft_encoder_state(
    *,
    rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
    base_parameters: PeftEncoderMaterializedState,
    lora_config: PeftEncoderTrainingBackendConfig,
    runtime_config: LoraClassifierModelRuntimeConfig,
    batch_size: int,
    seed: int,
    runtime_resource_cache: RuntimeResourceCache | None = None,
) -> dict[str, Any]:
    """materialized global PEFT encoder/head state로 labeled rows를 평가한다."""

    effective_labels = [str(label) for label in labels]
    if not effective_labels:
        raise ValueError("PEFT encoder evaluation labels must not be empty.")
    if batch_size <= 0:
        raise ValueError("PEFT encoder evaluation batch_size must be positive.")

    set_seed(int(seed))
    model, tokenizer = build_peft_encoder_text_classifier_from_config(
        labels=effective_labels,
        lora_config=lora_config,
        runtime_config=runtime_config,
        runtime_resource_cache=runtime_resource_cache,
    )
    load_peft_encoder_base_parameters_into_model(
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


def evaluate_peft_encoder_state_payload(
    *,
    rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
    base_parameters: PeftEncoderMaterializedState,
    lora_config: PeftEncoderTrainingBackendConfig,
    runtime_config: LoraClassifierModelRuntimeConfig,
    batch_size: int,
    seed: int,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    loss_kind: str = "cross_entropy_from_lora_classifier_logits",
    score_distribution_kind: str = LORA_CLASSIFIER_EVALUATION_DISTRIBUTION_KIND,
    selection_confidence_kind: str = LORA_CLASSIFIER_EVALUATION_CONFIDENCE_KIND,
) -> dict[str, object]:
    """PEFT encoder global state 평가 결과를 canonical payload로 반환한다."""

    report = evaluate_peft_encoder_state(
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
        loss_kind=loss_kind,
        score_distribution_kind=score_distribution_kind,
        selection_confidence_kind=selection_confidence_kind,
        mean_selection_confidence=float(report["mean_top_1_probability"]),
        mean_selection_margin=float(report["mean_margin_top1_top2"]),
    )


def evaluate_peft_encoder_validation_payload(
    *,
    rows: Sequence[LabeledQueryRow],
    adapter_state: object,
    base_parameters: PeftEncoderMaterializedState,
    objective_config: TrainingObjectiveConfig | None,
    runtime_config: LoraClassifierModelRuntimeConfig,
    batch_size: int,
    seed: int,
    runtime_resource_cache: RuntimeResourceCache | None = None,
) -> dict[str, object]:
    """FL validation runtime이 넘긴 PEFT-backed classifier state를 평가한다."""

    if not isinstance(adapter_state, LoraClassifierState | PeftClassifierState):
        raise ValueError(
            "PEFT-backed classifier evaluation requires classifier state; "
            f"got {type(adapter_state).__name__}."
        )
    is_peft_classifier = isinstance(adapter_state, PeftClassifierState)
    return evaluate_peft_encoder_state_payload(
        rows=rows,
        labels=adapter_state.label_schema,
        base_parameters=base_parameters,
        lora_config=_build_evaluation_training_backend_config(
            adapter_state=adapter_state,
            objective_config=objective_config,
        ),
        runtime_config=runtime_config,
        batch_size=batch_size,
        seed=seed,
        runtime_resource_cache=runtime_resource_cache,
        loss_kind=(
            "cross_entropy_from_peft_classifier_logits"
            if is_peft_classifier
            else "cross_entropy_from_lora_classifier_logits"
        ),
        score_distribution_kind=(
            PEFT_CLASSIFIER_EVALUATION_DISTRIBUTION_KIND
            if is_peft_classifier
            else LORA_CLASSIFIER_EVALUATION_DISTRIBUTION_KIND
        ),
        selection_confidence_kind=(
            PEFT_CLASSIFIER_EVALUATION_CONFIDENCE_KIND
            if is_peft_classifier
            else LORA_CLASSIFIER_EVALUATION_CONFIDENCE_KIND
        ),
    )


def evaluate_peft_encoder_simulation_validation_payload(
    *,
    rows: Sequence[LabeledQueryRow],
    adapter_state: object,
    aggregation_context: FederatedAggregationContext,
    objective_config: TrainingObjectiveConfig | None,
    runtime_config: LoraClassifierModelRuntimeConfig,
    batch_size: int,
    seed: int,
    scorer_backend_name: str,
    runtime_resource_cache: RuntimeResourceCache | None = None,
) -> dict[str, object]:
    """FL simulation이 넘긴 PEFT-backed classifier state를 평가한다."""

    state = require_peft_encoder_state(adapter_state)
    if scorer_backend_name not in PEFT_CLASSIFIER_ACCEPTED_EVALUATOR_NAMES:
        raise ValueError(
            "PEFT-backed classifier validation must use one of "
            f"{PEFT_CLASSIFIER_ACCEPTED_EVALUATOR_NAMES!r}: "
            f"{scorer_backend_name!r}."
        )
    return evaluate_peft_encoder_validation_payload(
        rows=rows,
        adapter_state=state,
        base_parameters=materialize_base_peft_encoder_state(
            base_state=state,
            context=aggregation_context,
        ),
        objective_config=objective_config,
        runtime_config=runtime_config,
        batch_size=batch_size,
        seed=seed,
        runtime_resource_cache=runtime_resource_cache,
    )


def require_peft_encoder_validation_backend(
    *,
    adapter_state: object,
    scorer_backend_name: str,
    prototype_scorer_backend_name: str,
) -> None:
    """PEFT encoder state에 prototype scorer validation이 붙는 drift를 막는다."""

    if not isinstance(adapter_state, LoraClassifierState | PeftClassifierState):
        return
    if scorer_backend_name in PEFT_CLASSIFIER_ACCEPTED_EVALUATOR_NAMES:
        return
    raise ValueError(
        "PEFT-backed classifier validation must use one of "
        f"{PEFT_CLASSIFIER_ACCEPTED_EVALUATOR_NAMES!r}. "
        f"{prototype_scorer_backend_name!r} is prototype/selection-only "
        "and does not read LoRA/classifier global state."
    )


def require_peft_encoder_state(
    adapter_state: object,
) -> LoraClassifierState | PeftClassifierState:
    """runtime adapter가 넘긴 shared state를 PEFT-backed classifier state로 검증한다."""

    if not isinstance(adapter_state, LoraClassifierState | PeftClassifierState):
        raise ValueError(
            "PEFT-backed classifier evaluation requires classifier state; "
            f"got {type(adapter_state).__name__}."
        )
    return adapter_state


def _build_evaluation_training_backend_config(
    *,
    adapter_state: LoraClassifierState | PeftClassifierState,
    objective_config: TrainingObjectiveConfig | None,
) -> PeftEncoderTrainingBackendConfig:
    if isinstance(adapter_state, PeftClassifierState):
        return build_peft_classifier_training_backend_config(objective_config)
    return build_lora_classifier_training_backend_config(objective_config)

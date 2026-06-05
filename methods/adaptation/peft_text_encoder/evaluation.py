"""PEFT text encoder global head state 평가 core."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
    build_peft_encoder_training_backend_config,
)
from methods.adaptation.peft_text_encoder.training.delta_extraction import (
    load_peft_encoder_base_parameters_into_model,
)
from methods.adaptation.peft_text_encoder.training.loops import (
    evaluate_classifier,
    set_seed,
)
from methods.adaptation.peft_text_encoder.training.modeling import (
    PeftEncoderModelRuntimeConfig,
    build_peft_text_encoder_with_linear_head_from_config,
)
from methods.adaptation.peft_text_encoder.training.pseudo_label_diagnostics import (
    tokenization_cache_namespace,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
    materialize_base_peft_encoder_state,
)
from methods.adaptation.query_text_views.data import build_dataloader
from methods.adaptation.query_text_views.tokenization import (
    resolve_text_tokenization_cache,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.evaluation.classification_payload import (
    build_classification_evaluation_payload,
)
from methods.federated.aggregation.base import FederatedAggregationContext
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierState,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

PEFT_ENCODER_CLASSIFIER_EVALUATOR_NAME = "peft_classifier_eval"
PEFT_ENCODER_ACCEPTED_CLASSIFIER_EVALUATOR_NAMES = (
    PEFT_ENCODER_CLASSIFIER_EVALUATOR_NAME,
)
PEFT_ENCODER_CLASSIFIER_DISTRIBUTION_KIND = "peft_classifier_logits_softmax"
PEFT_ENCODER_CLASSIFIER_CONFIDENCE_KIND = "peft_classifier_top1_probability"


def evaluate_peft_encoder_state(
    *,
    rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
    base_parameters: PeftEncoderMaterializedState,
    peft_config: PeftEncoderTrainingBackendConfig,
    runtime_config: PeftEncoderModelRuntimeConfig,
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
    model, tokenizer = build_peft_text_encoder_with_linear_head_from_config(
        labels=effective_labels,
        peft_config=peft_config,
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
    tokenization_cache = resolve_text_tokenization_cache(runtime_resource_cache)
    dataloader = build_dataloader(
        rows=list(rows),
        label_to_index=label_to_index,
        tokenizer=tokenizer,
        batch_size=int(batch_size),
        max_length=int(peft_config.max_length),
        task_prefix=peft_config.task_prefix,
        shuffle=False,
        tokenization_cache=tokenization_cache,
        tokenization_cache_namespace=tokenization_cache_namespace(peft_config),
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
    peft_config: PeftEncoderTrainingBackendConfig,
    runtime_config: PeftEncoderModelRuntimeConfig,
    batch_size: int,
    seed: int,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    loss_kind: str = "cross_entropy_from_peft_classifier_logits",
    score_distribution_kind: str = PEFT_ENCODER_CLASSIFIER_DISTRIBUTION_KIND,
    selection_confidence_kind: str = PEFT_ENCODER_CLASSIFIER_CONFIDENCE_KIND,
) -> dict[str, object]:
    """PEFT encoder global state 평가 결과를 canonical payload로 반환한다."""

    report = evaluate_peft_encoder_state(
        rows=rows,
        labels=labels,
        base_parameters=base_parameters,
        peft_config=peft_config,
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
    runtime_config: PeftEncoderModelRuntimeConfig,
    batch_size: int,
    seed: int,
    runtime_resource_cache: RuntimeResourceCache | None = None,
) -> dict[str, object]:
    """FL validation runtime이 넘긴 PEFT text encoder/head state를 평가한다."""

    if not isinstance(adapter_state, PeftClassifierState):
        raise ValueError(
            "PEFT text encoder/head evaluation requires PEFT encoder state; "
            f"got {type(adapter_state).__name__}."
        )
    return evaluate_peft_encoder_state_payload(
        rows=rows,
        labels=adapter_state.label_schema,
        base_parameters=base_parameters,
        peft_config=_build_evaluation_training_backend_config(
            adapter_state=adapter_state,
            objective_config=objective_config,
        ),
        runtime_config=runtime_config,
        batch_size=batch_size,
        seed=seed,
        runtime_resource_cache=runtime_resource_cache,
        loss_kind="cross_entropy_from_peft_classifier_logits",
        score_distribution_kind=PEFT_ENCODER_CLASSIFIER_DISTRIBUTION_KIND,
        selection_confidence_kind=PEFT_ENCODER_CLASSIFIER_CONFIDENCE_KIND,
    )


def evaluate_peft_encoder_simulation_validation_payload(
    *,
    rows: Sequence[LabeledQueryRow],
    adapter_state: object,
    aggregation_context: FederatedAggregationContext,
    objective_config: TrainingObjectiveConfig | None,
    runtime_config: PeftEncoderModelRuntimeConfig,
    batch_size: int,
    seed: int,
    scorer_backend_name: str,
    runtime_resource_cache: RuntimeResourceCache | None = None,
) -> dict[str, object]:
    """FL simulation이 넘긴 PEFT text encoder/head state를 평가한다."""

    state = require_peft_encoder_state(adapter_state)
    if scorer_backend_name not in PEFT_ENCODER_ACCEPTED_CLASSIFIER_EVALUATOR_NAMES:
        raise ValueError(
            "PEFT text encoder/head validation must use one of "
            f"{PEFT_ENCODER_ACCEPTED_CLASSIFIER_EVALUATOR_NAMES!r}: "
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

    if not isinstance(adapter_state, PeftClassifierState):
        return
    if scorer_backend_name in PEFT_ENCODER_ACCEPTED_CLASSIFIER_EVALUATOR_NAMES:
        return
    raise ValueError(
        "PEFT text encoder/head validation must use one of "
        f"{PEFT_ENCODER_ACCEPTED_CLASSIFIER_EVALUATOR_NAMES!r}. "
        f"{prototype_scorer_backend_name!r} is prototype/selection-only "
        "and does not read PEFT encoder/head global state."
    )


def require_peft_encoder_state(
    adapter_state: object,
) -> PeftClassifierState:
    """runtime adapter가 넘긴 shared state를 PEFT encoder state로 검증한다."""

    if not isinstance(adapter_state, PeftClassifierState):
        raise ValueError(
            "PEFT text encoder/head evaluation requires classifier state; "
            f"got {type(adapter_state).__name__}."
        )
    return adapter_state


def _build_evaluation_training_backend_config(
    *,
    adapter_state: PeftClassifierState,
    objective_config: TrainingObjectiveConfig | None,
) -> PeftEncoderTrainingBackendConfig:
    return build_peft_encoder_training_backend_config(objective_config)

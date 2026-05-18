"""Federated simulation용 validation 실행 adapter."""

from __future__ import annotations

from typing import Any

from methods.adaptation.lora_classifier.evaluation import (
    LORA_CLASSIFIER_EVALUATOR_NAME,
    evaluate_lora_classifier_validation_payload,
    require_lora_classifier_state,
    require_lora_classifier_validation_backend,
)
from methods.federated_ssl.runtime_fallbacks import (
    build_runtime_fallback_training_objective_config,
)
from methods.prototype.evaluation import build_prototype_candidate_evaluation_payload
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
)
from scripts.experiments.fl_ssl.federated_simulation.io.rows import parse_created_at
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedValidationConfig,
    SimulationEvaluation,
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_agent.scoring_runtime import (
    build_federated_scoring_service,
)
from scripts.runtime_adapters.federated_agent.selection_runtime import (
    select_federated_pseudo_labels,
)
from scripts.runtime_adapters.federated_agent.training_example_mapper import (
    build_federated_training_examples,
)
from scripts.runtime_adapters.federated_server.lora_classifier_state import (
    materialize_simulation_lora_classifier_base_state,
)
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter

PROTOTYPE_SIMILARITY_BACKEND_NAME = "prototype_similarity"


def build_training_examples(
    *,
    rows: list[LabeledQueryRow],
    adapter: EmbeddingAdapter,
    adapter_state: SharedAdapterState,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    scoring_service: Any,
    objective_config: TrainingObjectiveConfig | None = None,
) -> tuple[Any, ...]:
    """simulation row를 agent runtime training example builder로 변환한다."""
    return build_federated_training_examples(
        rows=rows,
        adapter=adapter,
        adapter_state=adapter_state,
        prototype_pack=prototype_pack,
        model_id=model_id,
        scoring_service=scoring_service,
        objective_config=objective_config,
        parse_created_at=parse_created_at,
    )


def evaluate_rows(
    *,
    rows: list[LabeledQueryRow],
    adapter: EmbeddingAdapter,
    adapter_state: SharedAdapterState,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    scoring_service: Any,
    confidence_threshold: float,
    margin_threshold: float,
    task_type: TrainingTaskType = TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,
    objective_config: TrainingObjectiveConfig | None = None,
) -> SimulationEvaluation:
    """validation row에 대해 top1 accuracy와 pseudo-label acceptance 비율을 계산한다."""
    effective_objective_config = _build_validation_objective_config(
        objective_config=objective_config,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )
    examples = build_training_examples(
        rows=rows,
        adapter=adapter,
        adapter_state=adapter_state,
        prototype_pack=prototype_pack,
        model_id=model_id,
        scoring_service=scoring_service,
        objective_config=effective_objective_config,
    )
    if not examples:
        return SimulationEvaluation(row_count=0, top1_accuracy=0.0, accepted_ratio=0.0)

    selection_result = select_federated_pseudo_labels(
        scored_events=tuple(example.evidence_scored_event for example in examples),
        training_task=_build_validation_task(
            model_id=model_id,
            model_revision=adapter_state.model_revision,
            training_scope=adapter_state.training_scope,
            task_type=task_type,
            objective_config=effective_objective_config,
        ),
    )
    true_labels = [str(row["mapped_label_4"]) for row in rows]
    predicted_labels = [candidate.label for candidate in selection_result.candidates]
    return SimulationEvaluation(
        **build_prototype_candidate_evaluation_payload(
            row_count=len(rows),
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            candidates=list(selection_result.candidates),
            evidences=list(selection_result.evidences),
            accepted_ratio=selection_result.accepted_ratio,
        )
    )


def build_validation_scoring_service(
    validation_config: FederatedValidationConfig,
    *,
    shared_state: SharedAdapterState | None = None,
) -> Any:
    """validation 설정으로 scoring service를 조립한다."""
    overrides: dict[str, str | int | float | bool] = {
        "scorer_backend_name": validation_config.scorer_backend_name,
    }
    if validation_config.score_policy_name is not None:
        overrides["score_policy_name"] = validation_config.score_policy_name
    if validation_config.score_top_k is not None:
        overrides["score_top_k"] = validation_config.score_top_k
    return build_federated_scoring_service(
        objective_config=build_runtime_fallback_training_objective_config(
            overrides=overrides
        ),
        similarity_name=validation_config.similarity_name,
        shared_state=shared_state,
    )


def evaluate_simulation_validation(
    *,
    request: SimulationRunRequest,
    adapter: EmbeddingAdapter,
    active: ActiveSimulationState,
    rows: list[LabeledQueryRow],
    objective_config: TrainingObjectiveConfig | None,
) -> SimulationEvaluation:
    """FL simulation request 기준 validation row를 평가한다."""

    if request.validation_config.scorer_backend_name == LORA_CLASSIFIER_EVALUATOR_NAME:
        return _evaluate_lora_classifier_validation(
            request=request,
            active=active,
            rows=rows,
            objective_config=objective_config,
        )
    require_lora_classifier_validation_backend(
        adapter_state=active.adapter_state,
        scorer_backend_name=request.validation_config.scorer_backend_name,
        prototype_scorer_backend_name=PROTOTYPE_SIMILARITY_BACKEND_NAME,
    )

    return evaluate_rows(
        rows=rows,
        adapter=adapter,
        adapter_state=active.adapter_state,
        prototype_pack=active.prototype_pack,
        model_id=request.model_id,
        scoring_service=build_validation_scoring_service(
            request.validation_config,
            shared_state=active.adapter_state,
        ),
        confidence_threshold=request.validation_config.confidence_threshold,
        margin_threshold=request.validation_config.margin_threshold,
        task_type=request.training_task_config.task_type,
        objective_config=objective_config,
    )


def _evaluate_lora_classifier_validation(
    *,
    request: SimulationRunRequest,
    active: ActiveSimulationState,
    rows: list[LabeledQueryRow],
    objective_config: TrainingObjectiveConfig | None,
) -> SimulationEvaluation:
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


def _build_validation_objective_config(
    *,
    objective_config: TrainingObjectiveConfig | None,
    confidence_threshold: float,
    margin_threshold: float,
) -> TrainingObjectiveConfig:
    overrides: dict[str, str | int | float | bool] = (
        {} if objective_config is None else objective_config.to_mapping()
    )
    overrides["confidence_threshold"] = confidence_threshold
    overrides["margin_threshold"] = margin_threshold
    return build_runtime_fallback_training_objective_config(overrides=overrides)


def _build_validation_task(
    *,
    model_id: str,
    model_revision: str,
    training_scope: str,
    task_type: TrainingTaskType,
    objective_config: TrainingObjectiveConfig,
) -> TrainingTask:
    return TrainingTask(
        schema_version="training_task.v1",
        task_id="simulation_validation_task",
        round_id="simulation_validation_round",
        model_id=model_id,
        model_revision=model_revision,
        task_type=task_type,
        training_scope=training_scope,
        local_epochs=1,
        batch_size=1,
        learning_rate=1e-4,
        max_steps=1,
        objective_config=objective_config,
        selection_policy=TrainingSelectionPolicy(),
    )

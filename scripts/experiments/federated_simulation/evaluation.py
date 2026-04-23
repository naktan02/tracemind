"""Federated simulation용 example build와 validation 계산."""

from __future__ import annotations

from agent.src.services.training.examples.service import (
    TrainingExampleBuildRequest,
    TrainingExampleService,
    TrainingExampleSource,
)
from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.training.backends.inputs.base import (
    WEAK_STRONG_PAIR_BACKEND_NAME,
)
from agent.src.services.training.selection.pseudo_label_service import (
    PseudoLabelSelectionService,
)
from agent.src.services.training.examples.models import (
    EmbeddedTrainingExample,
)
from scripts.labeled_query_rows import LabeledQueryRow
from shared.src.config.training_defaults import (
    DEFAULT_TRAINING_PROFILE,
    build_default_training_objective_config,
)
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter

from .io_utils import parse_created_at
from .models import FederatedValidationConfig, SimulationEvaluation


def build_training_examples(
    *,
    rows: list[LabeledQueryRow],
    adapter: EmbeddingAdapter,
    adapter_state: SharedAdapterState,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    scoring_service: ScoringService,
    objective_config: TrainingObjectiveConfig | None = None,
) -> tuple[EmbeddedTrainingExample, ...]:
    """simulation row를 agent runtime training example builder로 변환한다."""
    _validate_multiview_rows(
        rows=rows,
        objective_config=objective_config,
    )
    service = (
        TrainingExampleService()
        if objective_config is None
        else TrainingExampleService.from_objective_config(objective_config)
    )
    source_rows = tuple(
        _build_training_example_source(row)
        for row in rows
    )
    return service.build_examples(
        TrainingExampleBuildRequest(
            source_rows=source_rows,
            adapter=adapter,
            adapter_state=adapter_state,
            prototype_pack=prototype_pack,
            model_id=model_id,
            scoring_service=scoring_service,
        )
    )


def evaluate_rows(
    *,
    rows: list[LabeledQueryRow],
    adapter: EmbeddingAdapter,
    adapter_state: SharedAdapterState,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    scoring_service: ScoringService,
    confidence_threshold: float,
    margin_threshold: float,
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

    selection_result = PseudoLabelSelectionService().select(
        scored_events=tuple(
            example.evidence_scored_event for example in examples
        ),
        training_task=_build_validation_task(
            model_id=model_id,
            model_revision=adapter_state.model_revision,
            training_scope=adapter_state.training_scope,
            objective_config=effective_objective_config,
        ),
    )
    correct = sum(
        1
        for row, candidate in zip(rows, selection_result.candidates, strict=True)
        if candidate.label == str(row["mapped_label_4"])
    )

    return SimulationEvaluation(
        row_count=len(rows),
        top1_accuracy=correct / len(rows),
        accepted_ratio=selection_result.accepted_ratio,
    )


def build_validation_scoring_service(
    validation_config: FederatedValidationConfig,
    *,
    shared_state: SharedAdapterState | None = None,
) -> ScoringService:
    """validation 설정으로 scoring service를 조립한다."""
    overrides: dict[str, str | int | float | bool] = {
        "scorer_backend_name": validation_config.scorer_backend_name,
    }
    if validation_config.score_policy_name is not None:
        overrides["score_policy_name"] = validation_config.score_policy_name
    if validation_config.score_top_k is not None:
        overrides["score_top_k"] = validation_config.score_top_k
    return ScoringService.from_objective_config(
        build_default_training_objective_config(overrides=overrides),
        similarity_name=validation_config.similarity_name,
        shared_state=shared_state,
    )


def _build_training_example_source(row: LabeledQueryRow) -> TrainingExampleSource:
    return TrainingExampleSource(
        query_id=str(row["query_id"]),
        text=str(row["text"]),
        occurred_at=parse_created_at(str(row["created_at"])),
        weak_text=_optional_row_value(row, "weak_text"),
        strong_text=_optional_row_value(row, "strong_text"),
        weak_translated_text=_optional_row_value(row, "weak_translated_text"),
        strong_translated_text=_optional_row_value(row, "strong_translated_text"),
    )


def _optional_row_value(row: LabeledQueryRow, key: str) -> str | None:
    value = row.get(key)
    return None if value is None else str(value)


def _validate_multiview_rows(
    *,
    rows: list[LabeledQueryRow],
    objective_config: TrainingObjectiveConfig | None,
) -> None:
    backend_name = (
        DEFAULT_TRAINING_PROFILE.example_generation_backend_name
        if objective_config is None
        else (
            objective_config.example_generation_backend_name
            or DEFAULT_TRAINING_PROFILE.example_generation_backend_name
        )
    )
    if backend_name != WEAK_STRONG_PAIR_BACKEND_NAME:
        return
    for row in rows:
        if row.get("weak_text") and row.get("strong_text"):
            continue
        raise ValueError(
            "weak_strong_pair simulation requires each row to include both "
            "weak_text and strong_text."
        )


def _build_validation_objective_config(
    *,
    objective_config: TrainingObjectiveConfig | None,
    confidence_threshold: float,
    margin_threshold: float,
) -> TrainingObjectiveConfig:
    overrides: dict[str, str | int | float | bool] = (
        {}
        if objective_config is None
        else objective_config.to_mapping()
    )
    overrides["confidence_threshold"] = confidence_threshold
    overrides["margin_threshold"] = margin_threshold
    return build_default_training_objective_config(overrides=overrides)


def _build_validation_task(
    *,
    model_id: str,
    model_revision: str,
    training_scope: str,
    objective_config: TrainingObjectiveConfig,
) -> TrainingTask:
    return TrainingTask(
        schema_version="training_task.v1",
        task_id="simulation_validation_task",
        round_id="simulation_validation_round",
        model_id=model_id,
        model_revision=model_revision,
        task_type="pseudo_label_self_training",
        training_scope=training_scope,
        local_epochs=1,
        batch_size=1,
        learning_rate=1e-4,
        max_steps=1,
        objective_config=objective_config,
        selection_policy=TrainingSelectionPolicy(),
    )

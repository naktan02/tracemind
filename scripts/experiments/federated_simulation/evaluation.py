"""Federated simulation용 example build와 validation 계산."""

from __future__ import annotations

from typing import Any

from agent.src.services.federation.training_example_service import (
    TrainingExampleBuildRequest,
    TrainingExampleService,
    TrainingExampleSource,
)
from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.training.local_training_service import EmbeddedTrainingExample
from shared.src.contracts.adapter_contracts import VectorAdapterState
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import (
    build_default_training_objective_config,
)

from .io_utils import parse_created_at
from .models import FederatedValidationConfig, SimulationEvaluation


def build_training_examples(
    *,
    rows: list[dict[str, Any]],
    adapter: Any,
    adapter_state: VectorAdapterState,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    scoring_service: ScoringService,
) -> tuple[EmbeddedTrainingExample, ...]:
    """simulation row를 agent runtime training example builder로 변환한다."""
    service = TrainingExampleService()
    source_rows = tuple(
        TrainingExampleSource(
            query_id=str(row["query_id"]),
            text=str(row["text"]),
            occurred_at=parse_created_at(str(row["created_at"])),
        )
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
    rows: list[dict[str, Any]],
    adapter: Any,
    adapter_state: VectorAdapterState,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    scoring_service: ScoringService,
    confidence_threshold: float,
    margin_threshold: float,
) -> SimulationEvaluation:
    """validation row에 대해 top1 accuracy와 pseudo-label acceptance 비율을 계산한다."""
    examples = build_training_examples(
        rows=rows,
        adapter=adapter,
        adapter_state=adapter_state,
        prototype_pack=prototype_pack,
        model_id=model_id,
        scoring_service=scoring_service,
    )
    if not examples:
        return SimulationEvaluation(row_count=0, top1_accuracy=0.0, accepted_ratio=0.0)

    correct = 0
    accepted = 0
    for row, example in zip(rows, examples, strict=True):
        ranked_scores = sorted(
            example.scored_event.category_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked_scores[0][0] == str(row["mapped_label_4"]):
            correct += 1
        top_score = ranked_scores[0][1]
        runner_up_score = ranked_scores[1][1] if len(ranked_scores) > 1 else 0.0
        if (
            top_score >= confidence_threshold
            and (top_score - runner_up_score) >= margin_threshold
        ):
            accepted += 1

    return SimulationEvaluation(
        row_count=len(rows),
        top1_accuracy=correct / len(rows),
        accepted_ratio=accepted / len(rows),
    )


def build_validation_scoring_service(
    validation_config: FederatedValidationConfig,
) -> ScoringService:
    """validation 설정으로 scoring service를 조립한다."""
    return ScoringService.from_objective_config(
        build_default_training_objective_config(
            overrides={
                "score_policy_name": validation_config.score_policy_name,
                **(
                    {}
                    if validation_config.score_top_k is None
                    else {"score_top_k": validation_config.score_top_k}
                ),
            }
        ),
        similarity_name=validation_config.similarity_name,
    )

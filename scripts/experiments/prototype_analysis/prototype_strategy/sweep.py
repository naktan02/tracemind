"""Static threshold policy 비교 실행 로직."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from scripts.experiments.prototype_analysis.prototype_strategy.evaluation import (
    embed_rows,
    group_embeddings_by_label,
    score_embeddings,
)
from scripts.experiments.prototype_analysis.prototype_strategy.io_utils import dump_json
from scripts.experiments.prototype_analysis.prototype_strategy.models import (
    ThresholdArtifact,
    ThresholdPolicyEvaluation,
    ThresholdPolicyExperimentSummary,
)
from scripts.experiments.prototype_analysis.prototype_strategy.scoring import (
    PrototypeScoringConfigMixin,
)
from scripts.experiments.prototype_analysis.prototype_strategy.strategies import (
    build_requested_strategy,
)
from scripts.labeled_query_rows import LabeledQueryRow
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter

from .threshold_policies import StaticThresholdPolicy


@dataclass(slots=True)
class ThresholdPolicySelectionPolicy:
    """Validation 결과로 가장 적합한 threshold policy 후보를 고른다."""

    minimum_accepted_ratio: float = 0.5

    def select(
        self,
        evaluations: Sequence[ThresholdPolicyEvaluation],
    ) -> ThresholdPolicyEvaluation:
        if not evaluations:
            raise ValueError("At least one threshold policy evaluation is required.")
        eligible = tuple(
            evaluation
            for evaluation in evaluations
            if (
                evaluation.validation_metrics.accepted_ratio
                >= self.minimum_accepted_ratio
            )
        )
        candidates = eligible or tuple(evaluations)
        return max(candidates, key=self._selection_key)

    def sort(
        self,
        evaluations: Sequence[ThresholdPolicyEvaluation],
    ) -> tuple[ThresholdPolicyEvaluation, ...]:
        return tuple(
            sorted(
                evaluations,
                key=self._selection_key,
                reverse=True,
            )
        )

    @staticmethod
    def _selection_key(
        evaluation: ThresholdPolicyEvaluation,
    ) -> tuple[float | str, ...]:
        metrics = evaluation.validation_metrics
        return (
            metrics.accepted_accuracy,
            metrics.accepted_correct_ratio,
            metrics.accepted_ratio,
            _confidence_threshold_or_floor(evaluation.threshold_artifact),
            evaluation.policy_name,
            evaluation.candidate_name,
        )


@dataclass(slots=True, kw_only=True)
class ThresholdPolicyExperimentRunner(PrototypeScoringConfigMixin):
    """선택된 prototype 전략 위에서 static threshold policy를 비교한다."""

    selection_policy: ThresholdPolicySelectionPolicy = field(
        default_factory=ThresholdPolicySelectionPolicy
    )

    def run(
        self,
        request: "ThresholdPolicyExperimentRequest",
    ) -> ThresholdPolicyExperimentSummary:
        request.output_dir.mkdir(parents=True, exist_ok=True)

        train_embeddings = embed_rows(request.train_rows, request.adapter)
        validation_embeddings = embed_rows(request.validation_rows, request.adapter)
        test_embeddings = embed_rows(request.test_rows, request.adapter)

        embeddings_by_label = group_embeddings_by_label(
            rows=request.train_rows,
            embeddings=train_embeddings,
        )
        strategy = build_requested_strategy(
            strategy_name=request.strategy_name,
            seed=request.seed,
            kmeans_candidate_ks=request.kmeans_candidate_ks,
            kmeans_silhouette_sample_size=request.kmeans_silhouette_sample_size,
            dbscan_eps_values=request.dbscan_eps_values,
            dbscan_min_samples_values=request.dbscan_min_samples_values,
            dbscan_search_sample_size=request.dbscan_search_sample_size,
            dbscan_min_cluster_coverage=request.dbscan_min_cluster_coverage,
        )
        prototype_index = strategy.build(embeddings_by_label)
        dump_json(
            request.output_dir / "strategy" / "prototype_index.json",
            prototype_index.to_dict(),
        )

        scorer = self.build_prototype_index_scorer()
        validation_predictions = score_embeddings(
            rows=request.validation_rows,
            embeddings=validation_embeddings,
            prototype_index=prototype_index,
            scorer=scorer,
        )
        test_predictions = score_embeddings(
            rows=request.test_rows,
            embeddings=test_embeddings,
            prototype_index=prototype_index,
            scorer=scorer,
        )
        categories = sorted(prototype_index.categories.keys())

        evaluations: list[ThresholdPolicyEvaluation] = []
        for policy in request.threshold_policies:
            evaluations.extend(
                policy.build_evaluations(
                    validation_predictions=validation_predictions,
                    test_predictions=test_predictions,
                    categories=categories,
                )
            )

        selected_evaluation = self.selection_policy.select(evaluations)
        sorted_evaluations = self.selection_policy.sort(evaluations)
        summary = ThresholdPolicyExperimentSummary(
            run_id=request.run_id,
            strategy_name=strategy.name,
            prototype_index=prototype_index,
            selected_evaluation=selected_evaluation,
            policy_evaluations=sorted_evaluations,
        )
        evaluation_payload = {
            "evaluations": [evaluation.to_dict() for evaluation in sorted_evaluations]
        }
        dump_json(
            request.output_dir / "validation" / "policy_evaluations.json",
            evaluation_payload,
        )
        dump_json(request.output_dir / "summary.json", summary.to_dict())
        return summary


@dataclass(slots=True, frozen=True)
class ThresholdPolicyExperimentRequest:
    """Threshold policy 비교 실험 입력 묶음."""

    train_rows: Sequence[LabeledQueryRow]
    validation_rows: Sequence[LabeledQueryRow]
    test_rows: Sequence[LabeledQueryRow]
    adapter: EmbeddingAdapter
    strategy_name: str
    seed: int
    kmeans_candidate_ks: tuple[int, ...]
    kmeans_silhouette_sample_size: int
    dbscan_eps_values: tuple[float, ...]
    dbscan_min_samples_values: tuple[int, ...]
    dbscan_search_sample_size: int
    dbscan_min_cluster_coverage: float
    threshold_policies: tuple[StaticThresholdPolicy, ...]
    output_dir: Path
    run_id: str


def render_sweep_summary(summary: ThresholdPolicyExperimentSummary) -> str:
    selected = summary.selected_evaluation
    validation = selected.validation_metrics
    test = selected.test_metrics
    lines = [
        f"strategy={summary.strategy_name}",
        f"selected_policy={selected.policy_name}",
        f"selected_candidate={selected.candidate_name}",
        _render_selected_threshold(selected.threshold_artifact),
        (
            "validation_selected: "
            f"accepted_count={validation.accepted_count}, "
            f"accepted_ratio={validation.accepted_ratio:.4f}, "
            f"accepted_accuracy={validation.accepted_accuracy:.4f}, "
            f"accepted_correct_ratio={validation.accepted_correct_ratio:.4f}"
        ),
        (
            "test_selected: "
            f"accepted_count={test.accepted_count}, "
            f"accepted_ratio={test.accepted_ratio:.4f}, "
            f"accepted_accuracy={test.accepted_accuracy:.4f}, "
            f"accepted_correct_ratio={test.accepted_correct_ratio:.4f}"
        ),
    ]
    return "\n".join(lines)


def _confidence_threshold_or_floor(artifact: ThresholdArtifact) -> float:
    value = artifact.parameters.get("confidence_threshold")
    if isinstance(value, (int, float)):
        return float(value)
    classwise = artifact.parameters.get("confidence_threshold_by_label")
    if isinstance(classwise, dict) and classwise:
        numeric_values = [
            float(item) for item in classwise.values() if isinstance(item, (int, float))
        ]
        if numeric_values:
            return sum(numeric_values) / len(numeric_values)
    return -1.0


def _render_selected_threshold(artifact: ThresholdArtifact) -> str:
    global_threshold = artifact.parameters.get("confidence_threshold")
    if isinstance(global_threshold, (int, float)):
        return f"selected_threshold={float(global_threshold):.3f}"

    classwise = artifact.parameters.get("confidence_threshold_by_label")
    if isinstance(classwise, dict) and classwise:
        ordered = ", ".join(
            f"{label}:{float(value):.3f}"
            for label, value in sorted(classwise.items())
            if isinstance(value, (int, float))
        )
        return f"selected_thresholds={ordered}"
    return "selected_threshold=unknown"

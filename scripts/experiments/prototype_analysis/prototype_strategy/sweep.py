"""Static threshold policy 비교 실행 로직."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from methods.prototype.thresholding.models import ThresholdArtifact
from methods.prototype.thresholding.policies import StaticThresholdPolicy
from methods.prototype.thresholding.selection import ThresholdPolicySelectionPolicy
from scripts.experiments.prototype_analysis.prototype_strategy.evaluation import (
    embed_rows,
    group_embeddings_by_label,
)
from scripts.experiments.prototype_analysis.prototype_strategy.models import (
    ThresholdPolicyExperimentSummary,
)
from scripts.experiments.prototype_analysis.prototype_strategy.scoring import (
    PrototypeScoringConfigMixin,
)
from scripts.experiments.prototype_analysis.prototype_strategy.strategies import (
    build_requested_strategy,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter

from .threshold_artifact_writer import write_threshold_policy_artifacts
from .threshold_policy_evaluator import evaluate_threshold_policies


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

        scorer = self.build_prototype_index_scorer()
        evaluations = evaluate_threshold_policies(
            validation_rows=request.validation_rows,
            validation_embeddings=validation_embeddings,
            test_rows=request.test_rows,
            test_embeddings=test_embeddings,
            prototype_index=prototype_index,
            threshold_policies=request.threshold_policies,
            scorer=scorer,
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
        write_threshold_policy_artifacts(
            output_dir=request.output_dir,
            summary=summary,
        )
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

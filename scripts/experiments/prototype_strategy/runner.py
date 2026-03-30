"""Prototype 전략 실험 실행 오케스트레이션."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.experiments.prototype_strategy.evaluation import (
    embed_rows,
    evaluate_embeddings,
    group_embeddings_by_label,
)
from scripts.experiments.prototype_strategy.io_utils import dump_json
from scripts.experiments.prototype_strategy.models import (
    ExperimentSummary,
    PrototypeBuildStrategy,
    PrototypeIndex,
    StrategyEvaluationReport,
)
from scripts.experiments.prototype_strategy.projection import ProjectionService
from scripts.experiments.prototype_strategy.strategies import (
    DbscanPrototypeStrategy,
    KMeansPrototypeStrategy,
    MultiPrototypeScorer,
    SinglePrototypeStrategy,
)


@dataclass(slots=True)
class StrategySelectionPolicy:
    """validation 결과로 최종 전략을 고른다."""

    def select(
        self,
        reports: Sequence[StrategyEvaluationReport],
    ) -> StrategyEvaluationReport:
        if not reports:
            raise ValueError("At least one strategy report is required.")
        return max(
            reports,
            key=lambda report: (
                report.validation_metrics.top1_accuracy,
                report.validation_metrics.accepted_ratio,
                -report.prototype_index.prototype_count,
            ),
        )


@dataclass(slots=True)
class PrototypeExperimentRunner:
    """train/validation/test 흐름 전체를 오케스트레이션한다."""

    projection_service: ProjectionService
    confidence_threshold: float
    margin_threshold: float
    selection_policy: StrategySelectionPolicy = field(
        default_factory=StrategySelectionPolicy
    )

    def run(self, request: "PrototypeExperimentRequest") -> ExperimentSummary:
        request.output_dir.mkdir(parents=True, exist_ok=True)

        train_embeddings = embed_rows(request.train_rows, request.adapter)
        validation_embeddings = embed_rows(request.validation_rows, request.adapter)
        test_embeddings = embed_rows(request.test_rows, request.adapter)

        embeddings_by_label = group_embeddings_by_label(
            rows=request.train_rows,
            embeddings=train_embeddings,
        )
        scorer = MultiPrototypeScorer()
        reports: list[StrategyEvaluationReport] = []
        for strategy in request.strategies:
            prototype_index = strategy.build(embeddings_by_label)
            validation_metrics = evaluate_embeddings(
                rows=request.validation_rows,
                embeddings=validation_embeddings,
                prototype_index=prototype_index,
                confidence_threshold=self.confidence_threshold,
                margin_threshold=self.margin_threshold,
                scorer=scorer,
            )
            report = StrategyEvaluationReport(
                strategy_name=strategy.name,
                prototype_index=prototype_index,
                validation_metrics=validation_metrics,
            )
            reports.append(report)
            dump_json(
                request.output_dir
                / "strategies"
                / f"{strategy.name}.prototype_index.json",
                prototype_index.to_dict(),
            )
            dump_json(
                request.output_dir / "validation" / f"{strategy.name}.json",
                report.to_dict(),
            )

        selected_report = self.selection_policy.select(reports)
        projection_index = self._find_projection_index(
            reports,
            strategy_name="kmeans",
        )
        if projection_index is None:
            projection_index = selected_report.prototype_index
        projection_artifacts = self.projection_service.create(
            rows=request.train_rows,
            embeddings=train_embeddings,
            reducers=request.projection_reducers,
            output_dir=request.output_dir / "projections",
            prototype_index=projection_index,
        )
        test_metrics = evaluate_embeddings(
            rows=request.test_rows,
            embeddings=test_embeddings,
            prototype_index=selected_report.prototype_index,
            confidence_threshold=self.confidence_threshold,
            margin_threshold=self.margin_threshold,
            scorer=scorer,
        )
        summary = ExperimentSummary(
            run_id=request.run_id,
            selected_strategy=selected_report.strategy_name,
            strategy_reports=tuple(reports),
            test_metrics=test_metrics,
            projection_artifacts=projection_artifacts,
        )
        dump_json(request.output_dir / "summary.json", summary.to_dict())
        return summary

    @staticmethod
    def _find_projection_index(
        reports: Sequence[StrategyEvaluationReport],
        *,
        strategy_name: str,
    ) -> PrototypeIndex | None:
        for report in reports:
            if report.strategy_name == strategy_name:
                return report.prototype_index
        return None


@dataclass(slots=True, frozen=True)
class PrototypeExperimentRequest:
    """Prototype 전략 비교 실험 입력 묶음."""

    strategies: tuple[PrototypeBuildStrategy, ...]
    train_rows: Sequence[dict[str, Any]]
    validation_rows: Sequence[dict[str, Any]]
    test_rows: Sequence[dict[str, Any]]
    adapter: Any
    output_dir: Path
    run_id: str
    projection_reducers: tuple[str, ...]


def render_validation_summary(summary: ExperimentSummary) -> str:
    """터미널 출력용 요약 문자열을 만든다."""
    lines = [f"selected_strategy={summary.selected_strategy}"]
    for report in summary.strategy_reports:
        metrics = report.validation_metrics
        lines.append(
            f"{report.strategy_name}: "
            f"accuracy={metrics.top1_accuracy:.4f}, "
            f"accepted_ratio={metrics.accepted_ratio:.4f}, "
            f"prototype_count={report.prototype_index.prototype_count}"
        )
    lines.append(
        f"test: accuracy={summary.test_metrics.top1_accuracy:.4f}, "
        f"accepted_ratio={summary.test_metrics.accepted_ratio:.4f}"
    )
    return "\n".join(lines)


def build_strategy(
    *,
    strategy_name: str,
    seed: int,
    kmeans_candidate_ks: tuple[int, ...],
    kmeans_silhouette_sample_size: int,
    dbscan_eps_values: tuple[float, ...],
    dbscan_min_samples_values: tuple[int, ...],
    dbscan_search_sample_size: int,
    dbscan_min_cluster_coverage: float,
) -> PrototypeBuildStrategy:
    normalized_name = strategy_name.lower()
    if normalized_name == "single":
        return SinglePrototypeStrategy()
    if normalized_name == "kmeans":
        return KMeansPrototypeStrategy(
            candidate_ks=kmeans_candidate_ks,
            silhouette_sample_size=kmeans_silhouette_sample_size,
            random_state=seed,
        )
    if normalized_name == "dbscan":
        return DbscanPrototypeStrategy(
            eps_values=dbscan_eps_values,
            min_samples_values=dbscan_min_samples_values,
            search_sample_size=dbscan_search_sample_size,
            min_cluster_coverage=dbscan_min_cluster_coverage,
            random_state=seed,
        )
    raise ValueError(f"Unsupported strategy: {strategy_name}")


def build_strategies(
    *,
    strategy_name: str,
    seed: int,
    kmeans_candidate_ks: tuple[int, ...],
    kmeans_silhouette_sample_size: int,
    dbscan_eps_values: tuple[float, ...],
    dbscan_min_samples_values: tuple[int, ...],
    dbscan_search_sample_size: int,
    dbscan_min_cluster_coverage: float,
) -> tuple[PrototypeBuildStrategy, ...]:
    normalized_name = strategy_name.lower()
    if normalized_name == "all":
        return (
            SinglePrototypeStrategy(),
            KMeansPrototypeStrategy(
                candidate_ks=kmeans_candidate_ks,
                silhouette_sample_size=kmeans_silhouette_sample_size,
                random_state=seed,
            ),
            DbscanPrototypeStrategy(
                eps_values=dbscan_eps_values,
                min_samples_values=dbscan_min_samples_values,
                search_sample_size=dbscan_search_sample_size,
                min_cluster_coverage=dbscan_min_cluster_coverage,
                random_state=seed,
            ),
        )
    return (
        build_strategy(
            strategy_name=normalized_name,
            seed=seed,
            kmeans_candidate_ks=kmeans_candidate_ks,
            kmeans_silhouette_sample_size=kmeans_silhouette_sample_size,
            dbscan_eps_values=dbscan_eps_values,
            dbscan_min_samples_values=dbscan_min_samples_values,
            dbscan_search_sample_size=dbscan_search_sample_size,
            dbscan_min_cluster_coverage=dbscan_min_cluster_coverage,
        ),
    )

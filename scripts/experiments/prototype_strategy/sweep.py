"""Threshold sweep 실행 로직."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from scripts.experiments.prototype_strategy.evaluation import (
    embed_rows,
    evaluate_scored_predictions,
    group_embeddings_by_label,
    score_embeddings,
)
from scripts.experiments.prototype_strategy.io_utils import dump_json
from scripts.experiments.prototype_strategy.models import (
    EvaluationMetrics,
    PrototypeIndex,
)
from scripts.experiments.prototype_strategy.strategies import (
    DbscanPrototypeStrategy,
    KMeansPrototypeStrategy,
    MultiPrototypeScorer,
    SinglePrototypeStrategy,
)


@dataclass(slots=True)
class ThresholdSweepPoint:
    confidence_threshold: float
    margin_threshold: float
    validation_metrics: EvaluationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "validation_metrics": asdict(self.validation_metrics),
        }


@dataclass(slots=True)
class ThresholdSweepSummary:
    run_id: str
    strategy_name: str
    prototype_index: PrototypeIndex
    selected_point: ThresholdSweepPoint
    validation_grid: tuple[ThresholdSweepPoint, ...]
    test_metrics: EvaluationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy_name": self.strategy_name,
            "prototype_index": self.prototype_index.to_dict(),
            "selected_point": self.selected_point.to_dict(),
            "validation_grid": [point.to_dict() for point in self.validation_grid],
            "test_metrics": asdict(self.test_metrics),
        }


@dataclass(slots=True)
class ThresholdSweepSelectionPolicy:
    """validation에서 가장 유용한 pseudo-label threshold를 선택한다."""

    def select(
        self,
        points: Sequence[ThresholdSweepPoint],
    ) -> ThresholdSweepPoint:
        if not points:
            raise ValueError("At least one threshold point is required.")
        return max(
            points,
            key=lambda point: (
                point.validation_metrics.accepted_correct_ratio,
                point.validation_metrics.accepted_accuracy,
                point.validation_metrics.accepted_ratio,
                point.confidence_threshold,
                point.margin_threshold,
            ),
        )


@dataclass(slots=True)
class ThresholdSweepRunner:
    """선택된 prototype 전략 위에서 threshold grid를 평가한다."""

    selection_policy: ThresholdSweepSelectionPolicy = field(
        default_factory=ThresholdSweepSelectionPolicy
    )

    def run(
        self,
        *,
        train_rows: Sequence[dict[str, Any]],
        validation_rows: Sequence[dict[str, Any]],
        test_rows: Sequence[dict[str, Any]],
        adapter: Any,
        strategy_name: str,
        seed: int,
        kmeans_candidate_ks: tuple[int, ...],
        kmeans_silhouette_sample_size: int,
        dbscan_eps_values: tuple[float, ...],
        dbscan_min_samples_values: tuple[int, ...],
        dbscan_search_sample_size: int,
        dbscan_min_cluster_coverage: float,
        confidence_thresholds: tuple[float, ...],
        margin_thresholds: tuple[float, ...],
        output_dir: Path,
        run_id: str,
    ) -> ThresholdSweepSummary:
        output_dir.mkdir(parents=True, exist_ok=True)

        train_embeddings = embed_rows(train_rows, adapter)
        validation_embeddings = embed_rows(validation_rows, adapter)
        test_embeddings = embed_rows(test_rows, adapter)

        embeddings_by_label = group_embeddings_by_label(
            rows=train_rows,
            embeddings=train_embeddings,
        )
        strategy = build_strategy(
            strategy_name=strategy_name,
            seed=seed,
            kmeans_candidate_ks=kmeans_candidate_ks,
            kmeans_silhouette_sample_size=kmeans_silhouette_sample_size,
            dbscan_eps_values=dbscan_eps_values,
            dbscan_min_samples_values=dbscan_min_samples_values,
            dbscan_search_sample_size=dbscan_search_sample_size,
            dbscan_min_cluster_coverage=dbscan_min_cluster_coverage,
        )
        prototype_index = strategy.build(embeddings_by_label)
        dump_json(output_dir / "strategy" / "prototype_index.json", prototype_index.to_dict())

        scorer = MultiPrototypeScorer()
        validation_predictions = score_embeddings(
            rows=validation_rows,
            embeddings=validation_embeddings,
            prototype_index=prototype_index,
            scorer=scorer,
        )
        test_predictions = score_embeddings(
            rows=test_rows,
            embeddings=test_embeddings,
            prototype_index=prototype_index,
            scorer=scorer,
        )
        categories = sorted(prototype_index.categories.keys())

        points: list[ThresholdSweepPoint] = []
        for confidence_threshold in confidence_thresholds:
            for margin_threshold in margin_thresholds:
                metrics = evaluate_scored_predictions(
                    scored_predictions=validation_predictions,
                    categories=categories,
                    confidence_threshold=confidence_threshold,
                    margin_threshold=margin_threshold,
                )
                points.append(
                    ThresholdSweepPoint(
                        confidence_threshold=confidence_threshold,
                        margin_threshold=margin_threshold,
                        validation_metrics=metrics,
                    )
                )

        selected_point = self.selection_policy.select(points)
        test_metrics = evaluate_scored_predictions(
            scored_predictions=test_predictions,
            categories=categories,
            confidence_threshold=selected_point.confidence_threshold,
            margin_threshold=selected_point.margin_threshold,
        )

        sorted_points = tuple(
            sorted(
                points,
                key=lambda point: (
                    point.validation_metrics.accepted_correct_ratio,
                    point.validation_metrics.accepted_accuracy,
                    point.validation_metrics.accepted_ratio,
                    point.confidence_threshold,
                    point.margin_threshold,
                ),
                reverse=True,
            )
        )
        summary = ThresholdSweepSummary(
            run_id=run_id,
            strategy_name=strategy.name,
            prototype_index=prototype_index,
            selected_point=selected_point,
            validation_grid=sorted_points,
            test_metrics=test_metrics,
        )
        dump_json(output_dir / "validation" / "grid.json", {"points": [p.to_dict() for p in sorted_points]})
        dump_json(output_dir / "summary.json", summary.to_dict())
        return summary


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
) -> Any:
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


def render_sweep_summary(summary: ThresholdSweepSummary) -> str:
    lines = [
        f"strategy={summary.strategy_name}",
        (
            "selected_thresholds="
            f"confidence>={summary.selected_point.confidence_threshold:.3f}, "
            f"margin>={summary.selected_point.margin_threshold:.3f}"
        ),
        (
            "validation_selected: "
            f"accepted_count={summary.selected_point.validation_metrics.accepted_count}, "
            f"accepted_ratio={summary.selected_point.validation_metrics.accepted_ratio:.4f}, "
            f"accepted_accuracy={summary.selected_point.validation_metrics.accepted_accuracy:.4f}, "
            f"accepted_correct_ratio="
            f"{summary.selected_point.validation_metrics.accepted_correct_ratio:.4f}"
        ),
        (
            "test_selected: "
            f"accepted_count={summary.test_metrics.accepted_count}, "
            f"accepted_ratio={summary.test_metrics.accepted_ratio:.4f}, "
            f"accepted_accuracy={summary.test_metrics.accepted_accuracy:.4f}, "
            f"accepted_correct_ratio={summary.test_metrics.accepted_correct_ratio:.4f}"
        ),
    ]
    return "\n".join(lines)

"""Prototype strategy experiment unit tests."""

from __future__ import annotations

import math

import numpy as np

from scripts.experiments.prototype_strategy_experiment import (
    DbscanPrototypeStrategy,
    EvaluationMetrics,
    KMeansPrototypeStrategy,
    PrototypeIndex,
    PrototypeVector,
    SinglePrototypeStrategy,
    StrategyEvaluationReport,
    StrategySelectionPolicy,
)


def _normalize(x: float, y: float) -> list[float]:
    norm = math.sqrt(x * x + y * y)
    return [x / norm, y / norm]


def _build_embeddings() -> dict[str, np.ndarray]:
    return {
        "anxiety": np.asarray(
            [
                _normalize(1.0, 0.0),
                _normalize(0.99, 0.03),
                _normalize(0.96, -0.08),
                _normalize(0.0, 1.0),
                _normalize(0.03, 0.99),
                _normalize(-0.08, 0.96),
            ],
            dtype=np.float64,
        ),
        "normal": np.asarray(
            [
                _normalize(-1.0, 0.0),
                _normalize(-0.99, 0.02),
                _normalize(-0.97, -0.05),
            ],
            dtype=np.float64,
        ),
    }


def test_single_strategy_builds_one_prototype_per_category() -> None:
    strategy = SinglePrototypeStrategy()

    prototype_index = strategy.build(_build_embeddings())

    assert prototype_index.strategy_name == "single"
    assert prototype_index.prototype_count == 2
    assert prototype_index.prototype_count_by_category() == {
        "anxiety": 1,
        "normal": 1,
    }


def test_kmeans_and_dbscan_build_multiple_prototypes_for_separable_clusters() -> None:
    embeddings = {"anxiety": _build_embeddings()["anxiety"]}

    kmeans_index = KMeansPrototypeStrategy(
        candidate_ks=(2, 3),
        silhouette_sample_size=6,
        random_state=42,
    ).build(embeddings)
    dbscan_index = DbscanPrototypeStrategy(
        eps_values=(0.05, 0.1, 0.2),
        min_samples_values=(2,),
        search_sample_size=6,
        min_cluster_coverage=0.9,
        random_state=42,
    ).build(embeddings)

    assert len(kmeans_index.categories["anxiety"]) >= 2
    assert len(dbscan_index.categories["anxiety"]) >= 2


def test_selection_policy_prefers_accuracy_then_acceptance_then_simplicity() -> None:
    policy = StrategySelectionPolicy()

    def report(
        *,
        name: str,
        accuracy: float,
        accepted_ratio: float,
        prototype_count: int,
    ) -> StrategyEvaluationReport:
        categories = {
            "anxiety": [
                PrototypeVector(
                    prototype_id=f"{name}:{index}",
                    centroid=[1.0, 0.0],
                    member_count=2,
                )
                for index in range(prototype_count)
            ]
        }
        return StrategyEvaluationReport(
            strategy_name=name,
            prototype_index=PrototypeIndex(
                strategy_name=name,
                categories=categories,
            ),
            validation_metrics=EvaluationMetrics(
                row_count=10,
                top1_accuracy=accuracy,
                accepted_ratio=accepted_ratio,
                mean_true_label_score=0.7,
                mean_top1_score=0.8,
                mean_margin_top1_top2=0.2,
                confusion_matrix={"anxiety": {"anxiety": 10}},
                per_category={"anxiety": {"support": 10}},
            ),
        )

    selected = policy.select(
        [
            report(
                name="single",
                accuracy=0.8,
                accepted_ratio=0.7,
                prototype_count=1,
            ),
            report(
                name="kmeans",
                accuracy=0.8,
                accepted_ratio=0.75,
                prototype_count=3,
            ),
            report(
                name="dbscan",
                accuracy=0.82,
                accepted_ratio=0.6,
                prototype_count=2,
            ),
        ]
    )

    assert selected.strategy_name == "dbscan"

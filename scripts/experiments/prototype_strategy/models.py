"""Prototype strategy 실험 도메인 모델."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(slots=True)
class PrototypeVector:
    """하나의 cluster/centroid prototype."""

    prototype_id: str
    centroid: list[float]
    member_count: int


@dataclass(slots=True)
class PrototypeIndex:
    """전략별 category -> prototypes 결과."""

    strategy_name: str
    categories: dict[str, list[PrototypeVector]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prototype_count(self) -> int:
        return sum(len(prototypes) for prototypes in self.categories.values())

    def prototype_count_by_category(self) -> dict[str, int]:
        return {
            category: len(prototypes)
            for category, prototypes in sorted(self.categories.items())
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "prototype_count": self.prototype_count,
            "prototype_count_by_category": self.prototype_count_by_category(),
            "metadata": self.metadata,
            "categories": {
                category: [asdict(prototype) for prototype in prototypes]
                for category, prototypes in sorted(self.categories.items())
            },
        }


@dataclass(slots=True)
class EvaluationMetrics:
    """전략 평가 요약."""

    row_count: int
    top1_accuracy: float
    accepted_ratio: float
    mean_true_label_score: float
    mean_top1_score: float
    mean_margin_top1_top2: float
    confusion_matrix: dict[str, dict[str, int]]
    per_category: dict[str, dict[str, float | int]]
    accepted_count: int = 0
    accepted_accuracy: float = 0.0
    accepted_correct_ratio: float = 0.0
    accepted_mean_top1_score: float = 0.0
    accepted_mean_margin_top1_top2: float = 0.0


@dataclass(slots=True, frozen=True)
class ScoredPrediction:
    """threshold 실험을 위해 row별 score를 보존한 값 객체."""

    actual_label: str
    predicted_label: str
    true_label_score: float
    top1_score: float
    top2_score: float
    margin_top1_top2: float
    is_correct: bool


@dataclass(slots=True)
class StrategyEvaluationReport:
    """전략별 validation 결과."""

    strategy_name: str
    prototype_index: PrototypeIndex
    validation_metrics: EvaluationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "prototype_index": self.prototype_index.to_dict(),
            "validation_metrics": asdict(self.validation_metrics),
        }


@dataclass(slots=True)
class ProjectionArtifact:
    """Projection 결과 파일 경로."""

    reducer_name: str
    points_path: Path
    figure_path: Path


@dataclass(slots=True)
class ExperimentSummary:
    """전체 실험 요약."""

    run_id: str
    selected_strategy: str
    strategy_reports: tuple[StrategyEvaluationReport, ...]
    test_metrics: EvaluationMetrics
    projection_artifacts: tuple[ProjectionArtifact, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "selected_strategy": self.selected_strategy,
            "strategy_reports": [report.to_dict() for report in self.strategy_reports],
            "test_metrics": asdict(self.test_metrics),
            "projection_artifacts": [
                {
                    "reducer_name": artifact.reducer_name,
                    "points_path": str(artifact.points_path),
                    "figure_path": str(artifact.figure_path),
                }
                for artifact in self.projection_artifacts
            ],
        }


class PrototypeBuildStrategy(Protocol):
    """전략별 prototype 생성 인터페이스."""

    name: str

    def build(
        self,
        embeddings_by_label: Mapping[str, np.ndarray],
    ) -> PrototypeIndex:
        """label별 임베딩으로 prototype index를 만든다."""

"""Prototype strategy 실험 도메인 모델."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from methods.prototype.index import PrototypeIndex
from methods.prototype.thresholding.models import (
    EvaluationMetrics,
    ThresholdPolicyEvaluation,
)


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
    prototype_strategy_name: str | None = None
    prototype_points_path: Path | None = None
    visual_center_points_path: Path | None = None


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
                    "prototype_strategy_name": artifact.prototype_strategy_name,
                    "prototype_points_path": (
                        str(artifact.prototype_points_path)
                        if artifact.prototype_points_path is not None
                        else None
                    ),
                    "visual_center_points_path": (
                        str(artifact.visual_center_points_path)
                        if artifact.visual_center_points_path is not None
                        else None
                    ),
                }
                for artifact in self.projection_artifacts
            ],
        }


@dataclass(slots=True)
class ThresholdPolicyExperimentSummary:
    """Threshold policy 비교 실험 요약."""

    run_id: str
    strategy_name: str
    prototype_index: PrototypeIndex
    selected_evaluation: ThresholdPolicyEvaluation
    policy_evaluations: tuple[ThresholdPolicyEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy_name": self.strategy_name,
            "prototype_index": self.prototype_index.to_dict(),
            "selected_evaluation": self.selected_evaluation.to_dict(),
            "policy_evaluations": [
                evaluation.to_dict() for evaluation in self.policy_evaluations
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

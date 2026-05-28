"""Prototype threshold policy value objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class EvaluationMetrics:
    """Prototype score evaluation summary."""

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
class PaperReference:
    """Threshold policy source metadata."""

    title: str
    url: str
    venue: str | None = None
    year: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "venue": self.venue,
            "year": self.year,
        }


@dataclass(slots=True, frozen=True)
class ThresholdArtifact:
    """Static threshold artifact that can be passed to runtime."""

    threshold_kind: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold_kind": self.threshold_kind,
            "parameters": dict(self.parameters),
        }


@dataclass(slots=True, frozen=True)
class ScoredPrediction:
    """Per-row score preserved for threshold re-evaluation."""

    actual_label: str
    predicted_label: str
    true_label_score: float
    top1_score: float
    top2_score: float
    margin_top1_top2: float
    is_correct: bool


@dataclass(slots=True)
class ThresholdPolicyEvaluation:
    """Evaluation result for one threshold policy candidate."""

    policy_name: str
    candidate_name: str
    source_paper: PaperReference
    selection_params: dict[str, Any]
    threshold_artifact: ThresholdArtifact
    validation_metrics: EvaluationMetrics
    test_metrics: EvaluationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "candidate_name": self.candidate_name,
            "source_paper": self.source_paper.to_dict(),
            "selection_params": dict(self.selection_params),
            "threshold_artifact": self.threshold_artifact.to_dict(),
            "validation_metrics": asdict(self.validation_metrics),
            "test_metrics": asdict(self.test_metrics),
        }

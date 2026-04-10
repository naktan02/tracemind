"""Agent inference service package."""

from .baseline_service import BaselineConfig, BaselineService
from .decision_service import DecisionEvaluation, DecisionService
from .scoring_backends import (
    ClassifierHeadLogitsScoringBackend,
    PrototypeSimilarityScoringBackend,
    ScoringBackend,
    build_scoring_backend,
    register_scoring_backend,
)
from .scoring_policies import (
    MaxCosineScorePolicy,
    PrototypeScorePolicy,
    TopKMeanCosineScorePolicy,
    build_prototype_score_policy,
)
from .scoring_service import ScoringService
from .time_series_service import TimeSeriesAccumulator, TimeSeriesConfig

__all__ = [
    "BaselineConfig",
    "BaselineService",
    "ClassifierHeadLogitsScoringBackend",
    "DecisionEvaluation",
    "DecisionService",
    "MaxCosineScorePolicy",
    "PrototypeSimilarityScoringBackend",
    "PrototypeScorePolicy",
    "ScoringBackend",
    "ScoringService",
    "TimeSeriesAccumulator",
    "TimeSeriesConfig",
    "TopKMeanCosineScorePolicy",
    "build_scoring_backend",
    "build_prototype_score_policy",
    "register_scoring_backend",
]

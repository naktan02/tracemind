"""Prototype strategy experiment thin entrypoint.

실제 구현은 `scripts.experiments.prototype_strategy` 패키지에 둔다.
"""

from __future__ import annotations

from scripts.experiments.prototype_strategy import (  # noqa: E402
    DbscanPrototypeStrategy,
    EvaluationMetrics,
    ExperimentSummary,
    KMeansPrototypeStrategy,
    MultiPrototypeScorer,
    ProjectionArtifact,
    PrototypeExperimentRunner,
    PrototypeIndex,
    PrototypeVector,
    SinglePrototypeStrategy,
    StrategyEvaluationReport,
    StrategySelectionPolicy,
    build_strategies,
    build_strategy,
    render_validation_summary,
)
from scripts.experiments.prototype_strategy.main import main  # noqa: E402

__all__ = [
    "DbscanPrototypeStrategy",
    "EvaluationMetrics",
    "ExperimentSummary",
    "KMeansPrototypeStrategy",
    "MultiPrototypeScorer",
    "ProjectionArtifact",
    "PrototypeExperimentRunner",
    "PrototypeIndex",
    "PrototypeVector",
    "SinglePrototypeStrategy",
    "StrategyEvaluationReport",
    "StrategySelectionPolicy",
    "build_strategies",
    "build_strategy",
    "main",
    "render_validation_summary",
]


if __name__ == "__main__":
    main()

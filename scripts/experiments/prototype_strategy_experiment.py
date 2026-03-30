"""Prototype strategy experiment thin entrypoint.

실제 구현은 `scripts.experiments.prototype_strategy` 패키지에 둔다.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

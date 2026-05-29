"""Threshold sweep 산출물 저장."""

from __future__ import annotations

from pathlib import Path

from scripts.experiments.prototype_analysis.prototype_strategy.io_utils import (
    dump_json,
)
from scripts.experiments.prototype_analysis.prototype_strategy.models import (
    ThresholdPolicyExperimentSummary,
)


def write_threshold_policy_artifacts(
    *,
    output_dir: Path,
    summary: ThresholdPolicyExperimentSummary,
) -> None:
    """Threshold sweep summary와 evaluation artifact를 JSON으로 저장한다."""

    dump_json(
        output_dir / "strategy" / "prototype_index.json",
        summary.prototype_index.to_dict(),
    )
    dump_json(
        output_dir / "validation" / "policy_evaluations.json",
        {
            "evaluations": [
                evaluation.to_dict() for evaluation in summary.policy_evaluations
            ]
        },
    )
    dump_json(output_dir / "summary.json", summary.to_dict())

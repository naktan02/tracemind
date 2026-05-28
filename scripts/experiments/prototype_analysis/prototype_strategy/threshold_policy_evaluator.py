"""Threshold policy 후보 평가 계산."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from methods.prototype.index import PrototypeIndex
from methods.prototype.thresholding.models import ThresholdPolicyEvaluation
from methods.prototype.thresholding.policies import StaticThresholdPolicy
from scripts.experiments.prototype_analysis.prototype_strategy.evaluation import (
    score_embeddings,
)
from scripts.experiments.prototype_analysis.prototype_strategy.scoring import (
    PrototypeIndexScorer,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


def evaluate_threshold_policies(
    *,
    validation_rows: Sequence[LabeledQueryRow],
    validation_embeddings: np.ndarray,
    test_rows: Sequence[LabeledQueryRow],
    test_embeddings: np.ndarray,
    prototype_index: PrototypeIndex,
    threshold_policies: Sequence[StaticThresholdPolicy],
    scorer: PrototypeIndexScorer,
) -> tuple[ThresholdPolicyEvaluation, ...]:
    """Prototype score를 만들고 threshold policy 후보별 평가를 계산한다."""

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
    categories = tuple(sorted(prototype_index.categories.keys()))

    evaluations: list[ThresholdPolicyEvaluation] = []
    for policy in threshold_policies:
        evaluations.extend(
            policy.build_evaluations(
                validation_predictions=validation_predictions,
                test_predictions=test_predictions,
                categories=categories,
            )
        )
    return tuple(evaluations)

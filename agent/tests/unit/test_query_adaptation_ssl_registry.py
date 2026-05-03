"""Query adaptation SSL registry unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent.src.services.training.ssl.hooks.pseudo_label_selection import (
    registry as pseudo_label_selection_registry,
)
from agent.src.services.training.ssl.hooks.pseudo_label_selection.base import (
    PseudoLabelSelectionConfig,
)
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PSEUDO_LABEL_EVIDENCE_V1,
    PseudoLabelEvidence,
)


def _build_evidence() -> PseudoLabelEvidence:
    return PseudoLabelEvidence(
        schema_version=PSEUDO_LABEL_EVIDENCE_V1,
        evidence_id="evidence:q1",
        source_event_ref="q1",
        occurred_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        label="anxiety",
        confidence=0.62,
        confidence_kind="prototype_similarity",
        margin=0.01,
        top1_label="anxiety",
        top1_score=0.62,
        top2_label="depression",
        top2_score=0.61,
        raw_scores={"anxiety": 0.62, "depression": 0.61, "normal": 0.1},
    )


def test_margin_threshold_algorithm_requires_margin_cutoff() -> None:
    selection_hook = pseudo_label_selection_registry.build_pseudo_label_selection_hook(
        "top1_margin_threshold"
    )

    decision = selection_hook.evaluate(
        evidence=_build_evidence(),
        config=PseudoLabelSelectionConfig(
            confidence_threshold=0.6,
            margin_threshold=0.02,
        ),
    )

    assert decision.accepted is False
    assert decision.confidence == pytest.approx(0.62)
    assert decision.margin == pytest.approx(0.01)


def test_fixed_confidence_algorithm_ignores_margin_cutoff() -> None:
    selection_hook = pseudo_label_selection_registry.build_pseudo_label_selection_hook(
        "top1_confidence_only"
    )

    decision = selection_hook.evaluate(
        evidence=_build_evidence(),
        config=PseudoLabelSelectionConfig(
            confidence_threshold=0.6,
            margin_threshold=0.99,
        ),
    )

    assert decision.accepted is True
    assert decision.label == "anxiety"
    assert decision.runner_up_label == "depression"

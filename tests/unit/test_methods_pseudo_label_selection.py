"""Reusable pseudo-label selection method tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from methods.ssl.pseudo_label_selection.base import PseudoLabelSelectionConfig
from methods.ssl.pseudo_label_selection.registry import (
    build_pseudo_label_selection_hook,
    build_pseudo_label_selection_method,
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


def test_margin_threshold_method_requires_margin_cutoff() -> None:
    selection_method = build_pseudo_label_selection_method("top1_margin_threshold")

    decision = selection_method.evaluate(
        evidence=_build_evidence(),
        config=PseudoLabelSelectionConfig(
            confidence_threshold=0.6,
            margin_threshold=0.02,
        ),
    )

    assert selection_method.method_name == "top1_margin_threshold"
    assert selection_method.hook_name == "top1_margin_threshold"
    assert decision.accepted is False
    assert decision.confidence == pytest.approx(0.62)
    assert decision.margin == pytest.approx(0.01)


def test_fixed_confidence_method_ignores_margin_cutoff() -> None:
    selection_method = build_pseudo_label_selection_method("top1_confidence_only")

    decision = selection_method.evaluate(
        evidence=_build_evidence(),
        config=PseudoLabelSelectionConfig(
            confidence_threshold=0.6,
            margin_threshold=0.99,
        ),
    )

    assert decision.accepted is True
    assert decision.label == "anxiety"
    assert decision.runner_up_label == "depression"


def test_legacy_hook_name_resolves_to_selection_method() -> None:
    selection_method = build_pseudo_label_selection_hook("top1_confidence_only")

    assert selection_method.method_name == "top1_confidence_only"

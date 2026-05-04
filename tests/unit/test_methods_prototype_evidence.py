"""Reusable prototype evidence method tests."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from methods.prototype.evidence.helpers import (
    build_ranked_evidence,
    rank_category_scores,
    softmax_distribution,
)
from shared.src.domain.entities.inference.events import ScoredEvent


def _scored_event() -> ScoredEvent:
    return ScoredEvent(
        query_id="q1",
        occurred_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        translated_text=None,
        embedding_model_id="tracemind-embed",
        translation_model_id=None,
        category_scores={
            "normal": 0.3,
            "anxiety": 0.8,
            "depression": 0.6,
        },
    )


def test_rank_category_scores_orders_by_score_then_label() -> None:
    ranked = rank_category_scores(
        {
            "normal": 0.2,
            "depression": 0.7,
            "anxiety": 0.7,
        }
    )

    assert ranked == [
        ("anxiety", 0.7),
        ("depression", 0.7),
        ("normal", 0.2),
    ]


def test_build_ranked_evidence_uses_top1_top2_margin_and_metadata() -> None:
    scored_event = _scored_event()
    evidence = build_ranked_evidence(
        scored_event=scored_event,
        ranked_scores=rank_category_scores(scored_event.category_scores),
        confidence_kind="prototype_similarity",
        view_kind="single_view",
        backend_name="prototype_similarity_evidence",
    )

    assert evidence.evidence_id == "evidence:q1"
    assert evidence.top1_label == "anxiety"
    assert evidence.top1_score == pytest.approx(0.8)
    assert evidence.top2_label == "depression"
    assert evidence.top2_score == pytest.approx(0.6)
    assert evidence.margin == pytest.approx(0.2)
    assert evidence.sample_weight == pytest.approx(0.8)
    assert evidence.metadata["evidence_backend_name"] == (
        "prototype_similarity_evidence"
    )
    assert evidence.metadata["translation_used"] is False


def test_softmax_distribution_normalizes_scores() -> None:
    distribution = softmax_distribution(
        {"a": 2.0, "b": 1.0, "c": 0.0},
        temperature=1.0,
    )

    assert sum(distribution.values()) == pytest.approx(1.0)
    assert distribution["a"] > distribution["b"] > distribution["c"]


def test_helpers_reject_empty_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least one category score"):
        rank_category_scores({})
    with pytest.raises(ValueError, match="temperature"):
        softmax_distribution({"a": 1.0}, temperature=0.0)
    assert math.isfinite(softmax_distribution({"a": 1000.0}, temperature=1.0)["a"])

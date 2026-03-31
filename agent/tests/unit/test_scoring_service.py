"""ScoringService unit tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from agent.src.services.inference.scoring_policies import TopKMeanCosineScorePolicy
from agent.src.services.inference.scoring_service import ScoringService


def _load_prototypes() -> dict[str, list[float]]:
    fixture_path = Path(__file__).with_name("fixtures") / "prototype_pack_v1.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    return {
        category: values["centroid"]
        for category, values in payload["categories"].items()
    }


def test_score_returns_cosine_similarity_for_each_category() -> None:
    service = ScoringService()

    scores = service.score([1.0, 0.0, 0.0], _load_prototypes())

    assert scores["anxiety"] == pytest.approx(1.0)
    assert scores["depression"] == pytest.approx(0.0)
    assert scores["suicidal"] == pytest.approx(-1.0)
    assert scores["normal"] == pytest.approx(1.0 / math.sqrt(2.0))


def test_score_uses_best_matching_prototype_for_multi_prototype_category() -> None:
    service = ScoringService()

    scores = service.score(
        [1.0, 0.0],
        {
            "alert": ([0.0, 1.0], [1.0, 0.0]),
            "safe": ([0.0, -1.0],),
        },
    )

    assert scores["alert"] == pytest.approx(1.0)
    assert scores["safe"] == pytest.approx(0.0)


def test_score_rejects_unsupported_similarity_metric() -> None:
    service = ScoringService(similarity_name="sigmoid")

    with pytest.raises(ValueError, match="Unsupported similarity metric"):
        service.score([1.0, 0.0], {"anxiety": [1.0, 0.0]})


def test_score_rejects_dimension_mismatch() -> None:
    service = ScoringService()

    with pytest.raises(ValueError, match="dimensions must match"):
        service.score([1.0, 0.0, 0.0], {"anxiety": [1.0, 0.0]})


def test_score_rejects_zero_norm_embedding() -> None:
    service = ScoringService()

    with pytest.raises(ValueError, match="Embedding vector norm must be non-zero"):
        service.score([0.0, 0.0], {"anxiety": [1.0, 0.0]})


def test_score_rejects_zero_norm_prototype() -> None:
    service = ScoringService()

    with pytest.raises(ValueError, match="Prototype vector norm must be non-zero"):
        service.score([1.0, 0.0], {"anxiety": [0.0, 0.0]})


def test_score_can_switch_to_top_k_mean_policy() -> None:
    service = ScoringService(policy=TopKMeanCosineScorePolicy(top_k=2))

    scores = service.score(
        [1.0, 0.0],
        {
            "alert": ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]),
        },
    )

    assert scores["alert"] == pytest.approx(0.5)

"""Prototype scoring method core tests."""

from __future__ import annotations

import pytest

from methods.prototype.scoring.policies import (
    MaxCosineScorePolicy,
    TopKMeanCosineScorePolicy,
    build_prototype_score_policy,
)
from methods.prototype.scoring.similarity import score_prototype_categories


def test_score_prototype_categories_uses_best_matching_prototype() -> None:
    scores = score_prototype_categories(
        embedding=[1.0, 0.0],
        prototypes={
            "alert": ([0.0, 1.0], [1.0, 0.0]),
            "safe": ([0.0, -1.0],),
        },
        policy=MaxCosineScorePolicy(),
    )

    assert scores["alert"] == pytest.approx(1.0)
    assert scores["safe"] == pytest.approx(0.0)


def test_score_prototype_categories_supports_top_k_mean_policy() -> None:
    scores = score_prototype_categories(
        embedding=[1.0, 0.0],
        prototypes={
            "alert": ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]),
        },
        policy=TopKMeanCosineScorePolicy(top_k=2),
    )

    assert scores["alert"] == pytest.approx(0.5)


def test_build_prototype_score_policy_returns_registered_policy() -> None:
    policy = build_prototype_score_policy("topk_mean_cosine", top_k=3)

    assert isinstance(policy, TopKMeanCosineScorePolicy)
    assert policy.top_k == 3


def test_score_prototype_categories_rejects_unsupported_similarity() -> None:
    with pytest.raises(ValueError, match="Unsupported similarity metric"):
        score_prototype_categories(
            embedding=[1.0, 0.0],
            prototypes={"alert": [1.0, 0.0]},
            policy=MaxCosineScorePolicy(),
            similarity_name="sigmoid",
        )


def test_score_prototype_categories_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions must match"):
        score_prototype_categories(
            embedding=[1.0, 0.0, 0.0],
            prototypes={"alert": [1.0, 0.0]},
            policy=MaxCosineScorePolicy(),
        )


def test_score_prototype_categories_rejects_zero_norm_embedding() -> None:
    with pytest.raises(ValueError, match="Embedding vector norm must be non-zero"):
        score_prototype_categories(
            embedding=[0.0, 0.0],
            prototypes={"alert": [1.0, 0.0]},
            policy=MaxCosineScorePolicy(),
        )


def test_score_prototype_categories_rejects_zero_norm_prototype() -> None:
    with pytest.raises(ValueError, match="Prototype vector norm must be non-zero"):
        score_prototype_categories(
            embedding=[1.0, 0.0],
            prototypes={"alert": [0.0, 0.0]},
            policy=MaxCosineScorePolicy(),
        )


def test_top_k_mean_policy_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        TopKMeanCosineScorePolicy(top_k=0)


def test_build_prototype_score_policy_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="Unsupported prototype score policy"):
        build_prototype_score_policy("unknown")

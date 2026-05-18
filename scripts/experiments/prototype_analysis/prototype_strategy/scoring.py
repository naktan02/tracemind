"""Prototype strategy 실험용 scorer adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from methods.prototype.scoring.base import PrototypeScorePolicy
from methods.prototype.scoring.policy_registry import build_prototype_score_policy
from methods.prototype.scoring.similarity import score_prototype_categories
from scripts.experiments.prototype_analysis.prototype_strategy.models import (
    PrototypeIndex,
)

DEFAULT_PROTOTYPE_SIMILARITY_NAME = "cosine"
PROTOTYPE_SIMILARITY_SCORER_BACKEND_NAME = "prototype_similarity"
DEFAULT_PROTOTYPE_SCORE_POLICY_NAME = "max_cosine"


class PrototypeIndexScorer(Protocol):
    """PrototypeIndex에 대해 category score를 계산하는 실험용 scorer 인터페이스."""

    def score(
        self,
        embedding: Sequence[float],
        prototype_index: PrototypeIndex,
    ) -> dict[str, float]:
        """임베딩과 prototype index로 category score dict를 계산한다."""


@dataclass(slots=True, frozen=True)
class PrototypeScoringConfig:
    """Prototype strategy 실험용 scorer runtime 설정."""

    similarity_name: str = DEFAULT_PROTOTYPE_SIMILARITY_NAME
    scorer_backend_name: str | None = PROTOTYPE_SIMILARITY_SCORER_BACKEND_NAME
    score_policy_name: str | None = DEFAULT_PROTOTYPE_SCORE_POLICY_NAME
    score_top_k: int | None = None

    def build_scorer(self) -> "ConfiguredPrototypeIndexScorer":
        return ConfiguredPrototypeIndexScorer(
            scorer_backend_name=self.scorer_backend_name,
            score_policy_name=self.score_policy_name,
            score_top_k=self.score_top_k,
            similarity_name=self.similarity_name,
        )


@dataclass(slots=True, kw_only=True)
class PrototypeScoringConfigMixin:
    """Prototype strategy scorer 설정 축을 공유하는 mixin."""

    similarity_name: str = DEFAULT_PROTOTYPE_SIMILARITY_NAME
    scorer_backend_name: str | None = PROTOTYPE_SIMILARITY_SCORER_BACKEND_NAME
    score_policy_name: str | None = DEFAULT_PROTOTYPE_SCORE_POLICY_NAME
    score_top_k: int | None = None

    def build_scoring_config(self) -> PrototypeScoringConfig:
        return PrototypeScoringConfig(
            similarity_name=self.similarity_name,
            scorer_backend_name=self.scorer_backend_name,
            score_policy_name=self.score_policy_name,
            score_top_k=self.score_top_k,
        )

    def build_prototype_index_scorer(self) -> PrototypeIndexScorer:
        return self.build_scoring_config().build_scorer()


@dataclass(slots=True)
class ConfiguredPrototypeIndexScorer:
    """methods prototype scoring core를 재사용하는 prototype index scorer."""

    scorer_backend_name: str | None = PROTOTYPE_SIMILARITY_SCORER_BACKEND_NAME
    score_policy_name: str | None = DEFAULT_PROTOTYPE_SCORE_POLICY_NAME
    score_top_k: int | None = None
    similarity_name: str = "cosine"
    score_policy: PrototypeScorePolicy = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_prototype_similarity_backend(self.scorer_backend_name)
        self.score_policy = build_prototype_score_policy(
            self.score_policy_name or DEFAULT_PROTOTYPE_SCORE_POLICY_NAME,
            top_k=self.score_top_k,
        )

    def score(
        self,
        embedding: Sequence[float],
        prototype_index: PrototypeIndex,
    ) -> dict[str, float]:
        return score_prototype_categories(
            embedding=embedding,
            prototypes=_prototype_mapping(prototype_index),
            policy=self.score_policy,
            similarity_name=self.similarity_name,
        )


@dataclass(slots=True)
class MaxCosinePrototypeIndexScorer:
    """기존 max cosine 기본값과 호환되는 scorer wrapper."""

    similarity_name: str = DEFAULT_PROTOTYPE_SIMILARITY_NAME
    delegate: ConfiguredPrototypeIndexScorer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.delegate = PrototypeScoringConfig(
            similarity_name=self.similarity_name
        ).build_scorer()

    def score(
        self,
        embedding: Sequence[float],
        prototype_index: PrototypeIndex,
    ) -> dict[str, float]:
        return self.delegate.score(embedding, prototype_index)


def build_prototype_index_scorer(
    *,
    config: PrototypeScoringConfig | None = None,
    scorer_backend_name: str | None = PROTOTYPE_SIMILARITY_SCORER_BACKEND_NAME,
    score_policy_name: str | None = DEFAULT_PROTOTYPE_SCORE_POLICY_NAME,
    score_top_k: int | None = None,
    similarity_name: str = DEFAULT_PROTOTYPE_SIMILARITY_NAME,
) -> PrototypeIndexScorer:
    """prototype scoring config로 prototype index scorer를 조립한다."""

    if config is None:
        config = PrototypeScoringConfig(
            similarity_name=similarity_name,
            scorer_backend_name=scorer_backend_name,
            score_policy_name=score_policy_name,
            score_top_k=score_top_k,
        )
    return config.build_scorer()


def _prototype_mapping(
    prototype_index: PrototypeIndex,
) -> Mapping[str, tuple[tuple[float, ...], ...]]:
    return {
        category: tuple(
            tuple(float(value) for value in prototype.centroid)
            for prototype in prototypes
        )
        for category, prototypes in prototype_index.categories.items()
    }


def _validate_prototype_similarity_backend(scorer_backend_name: str | None) -> None:
    resolved_name = scorer_backend_name or PROTOTYPE_SIMILARITY_SCORER_BACKEND_NAME
    normalized_name = resolved_name.strip().lower()
    if normalized_name != PROTOTYPE_SIMILARITY_SCORER_BACKEND_NAME:
        raise ValueError(
            "Prototype strategy scorer only supports "
            f"{PROTOTYPE_SIMILARITY_SCORER_BACKEND_NAME!r}; "
            f"got {scorer_backend_name!r}."
        )

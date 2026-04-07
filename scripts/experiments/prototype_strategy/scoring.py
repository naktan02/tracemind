"""Prototype strategy 실험용 scorer adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from agent.src.services.inference.scoring_service import ScoringService
from scripts.experiments.prototype_strategy.models import PrototypeIndex
from shared.src.contracts.training_contracts import (
    TrainingConfigScalar,
    build_default_training_objective_config,
)


class PrototypeIndexScorer(Protocol):
    """PrototypeIndex에 대해 category score를 계산하는 실험용 scorer 인터페이스."""

    def score(
        self,
        embedding: Sequence[float],
        prototype_index: PrototypeIndex,
    ) -> dict[str, float]:
        """임베딩과 prototype index로 category score dict를 계산한다."""


@dataclass(slots=True)
class ConfiguredPrototypeIndexScorer:
    """shared scoring service를 재사용하는 prototype index scorer."""

    scorer_backend_name: str = "prototype_similarity"
    score_policy_name: str = "max_cosine"
    score_top_k: int | None = None
    similarity_name: str = "cosine"
    scoring_service: ScoringService = field(init=False, repr=False)

    def __post_init__(self) -> None:
        overrides: dict[str, TrainingConfigScalar] = {
            "scorer_backend_name": self.scorer_backend_name,
            "score_policy_name": self.score_policy_name,
        }
        if self.score_top_k is not None:
            overrides["score_top_k"] = self.score_top_k
        self.scoring_service = ScoringService.from_objective_config(
            build_default_training_objective_config(overrides=overrides),
            similarity_name=self.similarity_name,
        )

    def score(
        self,
        embedding: Sequence[float],
        prototype_index: PrototypeIndex,
    ) -> dict[str, float]:
        return self.scoring_service.score(
            embedding,
            _prototype_mapping(prototype_index),
        )


@dataclass(slots=True)
class MaxCosinePrototypeIndexScorer:
    """기존 max cosine 기본값과 호환되는 scorer wrapper."""

    similarity_name: str = "cosine"
    delegate: ConfiguredPrototypeIndexScorer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.delegate = ConfiguredPrototypeIndexScorer(
            scorer_backend_name="prototype_similarity",
            score_policy_name="max_cosine",
            score_top_k=None,
            similarity_name=self.similarity_name,
        )

    def score(
        self,
        embedding: Sequence[float],
        prototype_index: PrototypeIndex,
    ) -> dict[str, float]:
        return self.delegate.score(embedding, prototype_index)


def build_prototype_index_scorer(
    *,
    scorer_backend_name: str = "prototype_similarity",
    score_policy_name: str = "max_cosine",
    score_top_k: int | None = None,
    similarity_name: str = "cosine",
) -> PrototypeIndexScorer:
    """shared scoring config로 prototype index scorer를 조립한다."""

    return ConfiguredPrototypeIndexScorer(
        scorer_backend_name=scorer_backend_name,
        score_policy_name=score_policy_name,
        score_top_k=score_top_k,
        similarity_name=similarity_name,
    )


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

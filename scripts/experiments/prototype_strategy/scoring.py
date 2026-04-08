"""Prototype strategy 실험용 scorer adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from agent.src.services.inference.scoring_service import ScoringService
from scripts.experiments.prototype_strategy.models import PrototypeIndex
from shared.src.config.training_defaults import (
    DEFAULT_TRAINING_PROFILE,
    build_default_training_objective_config,
)
from shared.src.contracts.training_contracts import TrainingConfigScalar

DEFAULT_PROTOTYPE_SIMILARITY_NAME = "cosine"


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
    scorer_backend_name: str = DEFAULT_TRAINING_PROFILE.scorer_backend_name
    score_policy_name: str = DEFAULT_TRAINING_PROFILE.score_policy_name
    score_top_k: int | None = None

    def to_training_objective_overrides(self) -> dict[str, TrainingConfigScalar]:
        overrides: dict[str, TrainingConfigScalar] = {
            "scorer_backend_name": self.scorer_backend_name,
            "score_policy_name": self.score_policy_name,
        }
        if self.score_top_k is not None:
            overrides["score_top_k"] = self.score_top_k
        return overrides

    def build_scorer(self) -> "ConfiguredPrototypeIndexScorer":
        return ConfiguredPrototypeIndexScorer(
            scorer_backend_name=self.scorer_backend_name,
            score_policy_name=self.score_policy_name,
            score_top_k=self.score_top_k,
            similarity_name=self.similarity_name,
        )


@dataclass(slots=True, kw_only=True)
class PrototypeScoringRuntimeMixin:
    """Prototype strategy scorer 설정 축을 공유하는 mixin."""

    similarity_name: str = DEFAULT_PROTOTYPE_SIMILARITY_NAME
    scorer_backend_name: str = DEFAULT_TRAINING_PROFILE.scorer_backend_name
    score_policy_name: str = DEFAULT_TRAINING_PROFILE.score_policy_name
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
    """shared scoring service를 재사용하는 prototype index scorer."""

    scorer_backend_name: str = DEFAULT_TRAINING_PROFILE.scorer_backend_name
    score_policy_name: str = DEFAULT_TRAINING_PROFILE.score_policy_name
    score_top_k: int | None = None
    similarity_name: str = "cosine"
    scoring_service: ScoringService = field(init=False, repr=False)

    def __post_init__(self) -> None:
        config = PrototypeScoringConfig(
            similarity_name=self.similarity_name,
            scorer_backend_name=self.scorer_backend_name,
            score_policy_name=self.score_policy_name,
            score_top_k=self.score_top_k,
        )
        self.scoring_service = ScoringService.from_objective_config(
            build_default_training_objective_config(
                overrides=config.to_training_objective_overrides()
            ),
            similarity_name=config.similarity_name,
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
    scorer_backend_name: str = DEFAULT_TRAINING_PROFILE.scorer_backend_name,
    score_policy_name: str = DEFAULT_TRAINING_PROFILE.score_policy_name,
    score_top_k: int | None = None,
    similarity_name: str = DEFAULT_PROTOTYPE_SIMILARITY_NAME,
) -> PrototypeIndexScorer:
    """shared scoring config로 prototype index scorer를 조립한다."""

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

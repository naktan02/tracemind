"""Prototype-similarity scoring backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from methods.federated_ssl.runtime_fallbacks import RUNTIME_FALLBACK_TRAINING_PROFILE
from methods.prototype.scoring.base import PrototypeScorePolicy
from methods.prototype.scoring.policy_registry import build_prototype_score_policy
from methods.prototype.scoring.score_policies.max_cosine import MaxCosineScorePolicy
from methods.prototype.scoring.similarity import score_prototype_categories
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState

from .base import (
    PROTOTYPE_SIMILARITY_BACKEND_NAME,
    PROTOTYPE_SIMILARITY_CONFIDENCE_KIND,
    ScoringAssets,
    ScoringBackend,
)
from .registry import register_scoring_backend

PROTOTYPE_SIMILARITY_SCORING_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name=PROTOTYPE_SIMILARITY_BACKEND_NAME,
    display_name=PROTOTYPE_SIMILARITY_BACKEND_NAME,
    implementation_module=(
        "agent.src.services.inference.scoring_backends.prototype_similarity"
    ),
    core_method_name=PROTOTYPE_SIMILARITY_BACKEND_NAME,
    family_name="scoring",
    supported_adapter_kinds=("*",),
    metadata={
        "requires_shared_state": False,
        "confidence_kind": PROTOTYPE_SIMILARITY_CONFIDENCE_KIND,
    },
)


@dataclass(slots=True)
class PrototypeSimilarityScoringBackend:
    """prototype similarity를 계산하고 policy로 category score를 접는다."""

    similarity_name: str = "cosine"
    policy: PrototypeScorePolicy = field(default_factory=MaxCosineScorePolicy)
    backend_name: str = PROTOTYPE_SIMILARITY_BACKEND_NAME
    confidence_kind: str = PROTOTYPE_SIMILARITY_CONFIDENCE_KIND
    supported_adapter_kinds: tuple[str, ...] = ("*",)
    requires_shared_state: bool = False

    def score(
        self,
        embedding: Sequence[float],
        scoring_assets: ScoringAssets,
        shared_state: SharedAdapterState | None = None,
    ) -> dict[str, float]:
        del shared_state
        return score_prototype_categories(
            embedding=embedding,
            prototypes=scoring_assets,
            policy=self.policy,
            similarity_name=self.similarity_name,
        )


@register_scoring_backend(
    PROTOTYPE_SIMILARITY_BACKEND_NAME,
    catalog_entry=PROTOTYPE_SIMILARITY_SCORING_BACKEND_CATALOG_ENTRY,
)
def build_prototype_similarity_scoring_backend(
    objective_config: TrainingObjectiveConfig,
    similarity_name: str,
) -> ScoringBackend:
    """registry용 prototype similarity scoring backend factory."""

    policy_name = (
        objective_config.score_policy_name
        or RUNTIME_FALLBACK_TRAINING_PROFILE.score_policy_name
    )
    policy = build_prototype_score_policy(
        policy_name,
        top_k=objective_config.score_top_k,
    )
    return PrototypeSimilarityScoringBackend(
        similarity_name=similarity_name,
        policy=policy,
    )

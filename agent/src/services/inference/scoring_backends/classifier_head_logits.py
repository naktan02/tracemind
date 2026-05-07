"""Classifier-head logits scoring backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shared.src.contracts.adapter_contracts import ClassifierHeadState
from shared.src.contracts.adapter_family_metadata import CLASSIFIER_HEAD_FAMILY_METADATA
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState

from .base import (
    CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    CLASSIFIER_HEAD_LOGITS_CONFIDENCE_KIND,
    ScoringBackend,
)
from .registry import register_scoring_backend

CLASSIFIER_HEAD_LOGITS_SCORING_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name=CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    display_name=CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    implementation_module=(
        "agent.src.services.inference.scoring_backends.classifier_head_logits"
    ),
    core_method_name=CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    family_name="scoring",
    supported_adapter_kinds=(CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,),
    tags=("requires_shared_state",),
    metadata={
        "requires_shared_state": True,
        "confidence_kind": CLASSIFIER_HEAD_LOGITS_CONFIDENCE_KIND,
    },
)


@dataclass(slots=True)
class ClassifierHeadLogitsScoringBackend:
    """공통 classifier head state로 category logits를 계산한다."""

    backend_name: str = CLASSIFIER_HEAD_LOGITS_BACKEND_NAME
    confidence_kind: str = CLASSIFIER_HEAD_LOGITS_CONFIDENCE_KIND
    supported_adapter_kinds: tuple[str, ...] = ("classifier_head",)
    requires_shared_state: bool = True

    def score(
        self,
        embedding: Sequence[float],
        prototypes: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
        shared_state: SharedAdapterState | None = None,
    ) -> dict[str, float]:
        del prototypes
        if not isinstance(shared_state, ClassifierHeadState):
            raise ValueError(
                "classifier_head_logits backend requires "
                "ClassifierHeadState as shared_state."
            )
        embedding_vector = _coerce_embedding_vector(embedding)
        return shared_state.compute_logits(embedding_vector)


@register_scoring_backend(
    CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    catalog_entry=CLASSIFIER_HEAD_LOGITS_SCORING_BACKEND_CATALOG_ENTRY,
)
def build_classifier_head_logits_scoring_backend(
    objective_config: TrainingObjectiveConfig,
    similarity_name: str,
) -> ScoringBackend:
    """registry용 classifier-head logits scoring backend factory."""

    del objective_config, similarity_name
    return ClassifierHeadLogitsScoringBackend()


def _coerce_embedding_vector(values: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError("embedding must not be empty.")
    return vector

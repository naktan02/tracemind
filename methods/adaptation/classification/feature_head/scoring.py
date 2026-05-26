"""Classifier-head scoring backend core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from methods.adaptation.scoring_registry import register_shared_adapter_scoring_backend
from shared.src.contracts.adapter_contract_families.classifier_head import (
    CLASSIFIER_HEAD_ADAPTER_KIND,
    ClassifierHeadState,
)
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState

CLASSIFIER_HEAD_LOGITS_BACKEND_NAME = "classifier_head_logits"
CLASSIFIER_HEAD_LOGITS_CONFIDENCE_KIND = "classifier_head_logit_top1"

CLASSIFIER_HEAD_LOGITS_SCORING_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name=CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    display_name=CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    implementation_module="methods.adaptation.classification.feature_head.scoring",
    core_method_name=CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    family_name="scoring",
    supported_adapter_kinds=(CLASSIFIER_HEAD_ADAPTER_KIND,),
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
    supported_adapter_kinds: tuple[str, ...] = (CLASSIFIER_HEAD_ADAPTER_KIND,)
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


@register_shared_adapter_scoring_backend(
    CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    catalog_entry=CLASSIFIER_HEAD_LOGITS_SCORING_BACKEND_CATALOG_ENTRY,
)
def build_classifier_head_logits_scoring_backend(
    objective_config: TrainingObjectiveConfig,
    similarity_name: str,
) -> ClassifierHeadLogitsScoringBackend:
    """classifier-head logits scoring backend factory."""

    del objective_config, similarity_name
    return ClassifierHeadLogitsScoringBackend()


def _coerce_embedding_vector(values: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError("embedding must not be empty.")
    return vector

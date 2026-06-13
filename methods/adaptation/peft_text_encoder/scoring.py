"""PEFT text encoder classifier-head scoring backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from methods.adaptation.scoring_backend import ScoringAssets
from methods.adaptation.scoring_registry import register_shared_adapter_scoring_backend
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
    PeftClassifierAdapterStatePayload,
)
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.scoring_contracts import ScoringConfigPayload
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState

PEFT_CLASSIFIER_HEAD_LOGITS_BACKEND_NAME = "peft_classifier_head_logits"
PEFT_CLASSIFIER_HEAD_WEIGHTS_ASSET_KEY = "classifier_head_weights"
PEFT_CLASSIFIER_HEAD_BIASES_ASSET_KEY = "classifier_head_biases"

PEFT_CLASSIFIER_HEAD_LOGITS_SCORING_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name=PEFT_CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    display_name=PEFT_CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    implementation_module="methods.adaptation.peft_text_encoder.scoring",
    core_method_name=PEFT_CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    family_name="scoring",
    supported_adapter_kinds=(PEFT_CLASSIFIER_ADAPTER_KIND,),
    tags=("requires_shared_state", "requires_scoring_assets"),
    metadata={
        "requires_shared_state": True,
        "required_assets": (
            PEFT_CLASSIFIER_HEAD_WEIGHTS_ASSET_KEY,
            PEFT_CLASSIFIER_HEAD_BIASES_ASSET_KEY,
        ),
    },
)


@dataclass(slots=True)
class PeftClassifierHeadLogitsScoringBackend:
    """PEFT-classifier state와 materialized linear head로 logits를 계산한다."""

    backend_name: str = PEFT_CLASSIFIER_HEAD_LOGITS_BACKEND_NAME
    supported_adapter_kinds: tuple[str, ...] = (PEFT_CLASSIFIER_ADAPTER_KIND,)
    requires_shared_state: bool = True

    def score(
        self,
        embedding: Sequence[float],
        scoring_assets: ScoringAssets,
        shared_state: SharedAdapterState | None = None,
    ) -> dict[str, float]:
        if not isinstance(shared_state, PeftClassifierAdapterStatePayload):
            raise ValueError(
                "peft_classifier_head_logits backend requires "
                "PeftClassifierAdapterStatePayload as shared_state."
            )
        labels = shared_state.labels
        embedding_vector = _coerce_vector(embedding, field_name="embedding")
        weights = _coerce_matrix(
            scoring_assets.get(PEFT_CLASSIFIER_HEAD_WEIGHTS_ASSET_KEY),
            field_name=PEFT_CLASSIFIER_HEAD_WEIGHTS_ASSET_KEY,
        )
        biases = _coerce_vector(
            scoring_assets.get(PEFT_CLASSIFIER_HEAD_BIASES_ASSET_KEY),
            field_name=PEFT_CLASSIFIER_HEAD_BIASES_ASSET_KEY,
        )
        if len(weights) != len(labels) or len(biases) != len(labels):
            raise ValueError("classifier head asset label dimension mismatch.")
        return {
            label: _dot(embedding_vector, weights[index]) + biases[index]
            for index, label in enumerate(labels)
        }


@register_shared_adapter_scoring_backend(
    PEFT_CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    catalog_entry=PEFT_CLASSIFIER_HEAD_LOGITS_SCORING_BACKEND_CATALOG_ENTRY,
)
def build_peft_classifier_head_logits_scoring_backend(
    scoring_config: ScoringConfigPayload,
    similarity_name: str,
) -> PeftClassifierHeadLogitsScoringBackend:
    """PEFT classifier-head logits scoring backend factory."""

    del scoring_config, similarity_name
    return PeftClassifierHeadLogitsScoringBackend()


def build_peft_classifier_head_scoring_assets(
    *,
    classifier_head_artifact: Mapping[str, object],
    label_schema: Sequence[str],
) -> dict[str, list[float] | list[list[float]]]:
    """server-published classifier-head artifact를 scoring assets로 정규화한다."""

    labels = tuple(str(label).strip() for label in label_schema)
    if not labels or any(not label for label in labels):
        raise ValueError("PEFT classifier head scoring requires label_schema.")
    raw_weights = classifier_head_artifact.get("classifier_head_weights")
    raw_biases = classifier_head_artifact.get("classifier_head_biases")
    if not isinstance(raw_weights, Mapping) or not isinstance(raw_biases, Mapping):
        raise ValueError(
            "classifier head artifact requires classifier_head_weights and "
            "classifier_head_biases objects."
        )
    return {
        PEFT_CLASSIFIER_HEAD_WEIGHTS_ASSET_KEY: [
            _coerce_vector(raw_weights.get(label), field_name=f"weights.{label}")
            for label in labels
        ],
        PEFT_CLASSIFIER_HEAD_BIASES_ASSET_KEY: [
            _coerce_scalar(raw_biases.get(label), field_name=f"biases.{label}")
            for label in labels
        ],
    }


def _coerce_matrix(value: object, *, field_name: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a vector sequence.")
    matrix = [_coerce_vector(row, field_name=field_name) for row in value]
    if not matrix:
        raise ValueError(f"{field_name} must not be empty.")
    return matrix


def _coerce_vector(value: object, *, field_name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a numeric sequence.")
    vector = [_coerce_scalar(item, field_name=field_name) for item in value]
    if not vector:
        raise ValueError(f"{field_name} must not be empty.")
    return vector


def _coerce_scalar(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field_name} must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be numeric.") from error


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimension does not match classifier head weights.")
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))

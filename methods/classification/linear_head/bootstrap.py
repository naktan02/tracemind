"""Classifier-head 초기화 helper."""

from __future__ import annotations

from datetime import datetime

from shared.src.contracts.adapter_contract_families.classifier_head import (
    ClassifierHeadState,
)
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    extract_category_centroids,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState


def build_classifier_head_state_from_prototype_pack(
    *,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    model_revision: str,
    training_scope: str,
    updated_at: datetime,
    logit_scale: float = 8.0,
) -> SharedAdapterState:
    """bootstrap prototype centroid로 classifier-head 초기 상태를 만든다."""

    centroids = extract_category_centroids(prototype_pack)
    if not centroids:
        raise ValueError(
            "Classifier-head initialization requires at least one centroid."
        )
    return ClassifierHeadState(
        schema_version="classifier_head_state.v1",
        adapter_kind="classifier_head",
        model_id=model_id,
        model_revision=model_revision,
        training_scope=training_scope,
        updated_at=updated_at,
        label_weights={
            label: [float(value) * logit_scale for value in centroid]
            for label, centroid in centroids.items()
        },
        label_biases={label: 0.0 for label in centroids},
    )


def build_zero_classifier_head_state(
    *,
    round_runtime_config: object,
    model_id: str,
    model_revision: str,
    training_scope: str,
    embedding_dim: int,
    labels: tuple[str, ...] | list[str],
    updated_at: datetime,
) -> SharedAdapterState:
    """linear-head update family의 zero-initialized shared state를 만든다."""

    _ = round_runtime_config
    return ClassifierHeadState.zero_initialized(
        model_id=model_id,
        model_revision=model_revision,
        labels=labels,
        embedding_dim=embedding_dim,
        training_scope=training_scope,
        updated_at=updated_at,
    )

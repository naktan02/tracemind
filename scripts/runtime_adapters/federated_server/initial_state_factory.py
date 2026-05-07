"""Initial shared-state factories for federated simulation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from shared.src.contracts.adapter_contracts import (
    ClassifierHeadState,
    LoraClassifierState,
    VectorAdapterState,
)
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    extract_category_centroids,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState


def build_initial_shared_state(
    *,
    round_runtime_config: Any,
    model_id: str,
    model_revision: str,
    training_scope: str,
    embedding_dim: int,
    labels: tuple[str, ...] | list[str],
    updated_at: datetime,
) -> SharedAdapterState:
    """simulation bootstrap용 초기 shared state를 family별로 만든다."""
    adapter_family_name = str(round_runtime_config.adapter_family_name).strip().lower()
    if adapter_family_name == "classifier_head":
        return ClassifierHeadState.zero_initialized(
            model_id=model_id,
            model_revision=model_revision,
            labels=labels,
            embedding_dim=embedding_dim,
            training_scope=training_scope,
            updated_at=updated_at,
        )
    if adapter_family_name == "lora_classifier":
        lora_config = getattr(round_runtime_config, "lora_classifier", None)
        if lora_config is None:
            raise ValueError(
                "lora_classifier round runtime requires lora_classifier "
                "bootstrap config."
            )
        return LoraClassifierState(
            model_id=model_id,
            model_revision=model_revision,
            training_scope=training_scope,
            updated_at=updated_at,
            backbone=lora_config.backbone_payload(),
            lora_config=lora_config.lora_config_payload(),
            label_schema=list(labels),
            lora_adapter_artifact_ref=lora_config.lora_adapter_artifact_ref,
            classifier_head_artifact_ref=lora_config.classifier_head_artifact_ref,
            artifact_format=lora_config.artifact_format,
        )
    return VectorAdapterState.identity(
        model_id=model_id,
        model_revision=model_revision,
        training_scope=training_scope,
        embedding_dim=embedding_dim,
        updated_at=updated_at,
    )


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

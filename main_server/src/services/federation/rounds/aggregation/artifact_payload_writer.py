"""Aggregate artifact payload 저장 helper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from torch import Tensor

from methods.federated.aggregation.base import (
    is_safetensors_aggregated_artifact,
    safetensors_aggregated_artifact_parts,
)

from .artifact_refs import AggregationArtifactStore


def save_aggregated_artifact_payload(
    *,
    artifact_store: AggregationArtifactStore,
    artifact_ref: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """aggregate artifact payload를 저장하고 JSON-safe summary를 반환한다."""

    if is_safetensors_aggregated_artifact(payload):
        tensors, metadata = safetensors_aggregated_artifact_parts(payload)
        artifact_store.save_safetensors_artifact_ref(
            artifact_ref=artifact_ref,
            tensors=cast(dict[str, Tensor], dict(tensors)),
            metadata=dict(metadata),
        )
        return {
            "artifact_format": "safetensors",
            "tensor_count": len(tensors),
            "metadata_keys": sorted(str(key) for key in metadata),
        }
    artifact_store.save_json_artifact_ref(
        artifact_ref=artifact_ref,
        payload=dict(payload),
    )
    return dict(payload)

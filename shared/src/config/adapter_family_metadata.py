"""Shared adapter family canonical 메타데이터."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.contracts.adapter_contracts import AdapterKind
from shared.src.contracts.training_contracts import UpdatePayloadFormat


@dataclass(frozen=True, slots=True)
class SharedAdapterFamilyMetadata:
    """agent/server가 함께 해석하는 adapter family 메타데이터."""

    family_name: str
    adapter_kind: str
    canonical_update_payload_format: str
    accepted_update_payload_formats: tuple[str, ...]


DIAGONAL_SCALE_FAMILY_METADATA = SharedAdapterFamilyMetadata(
    family_name=AdapterKind.DIAGONAL_SCALE.value,
    adapter_kind=AdapterKind.DIAGONAL_SCALE.value,
    canonical_update_payload_format=UpdatePayloadFormat.DIAGONAL_SCALE_UPDATE.value,
    accepted_update_payload_formats=(
        UpdatePayloadFormat.DIAGONAL_SCALE_UPDATE.value,
        UpdatePayloadFormat.LEGACY_VECTOR_ADAPTER_DELTA.value,
    ),
)

CLASSIFIER_HEAD_FAMILY_METADATA = SharedAdapterFamilyMetadata(
    family_name=AdapterKind.CLASSIFIER_HEAD.value,
    adapter_kind=AdapterKind.CLASSIFIER_HEAD.value,
    canonical_update_payload_format=UpdatePayloadFormat.CLASSIFIER_HEAD_UPDATE.value,
    accepted_update_payload_formats=(UpdatePayloadFormat.CLASSIFIER_HEAD_UPDATE.value,),
)

_SHARED_ADAPTER_FAMILY_METADATA_REGISTRY: tuple[SharedAdapterFamilyMetadata, ...] = (
    DIAGONAL_SCALE_FAMILY_METADATA,
    CLASSIFIER_HEAD_FAMILY_METADATA,
)


def list_shared_adapter_family_metadata() -> tuple[SharedAdapterFamilyMetadata, ...]:
    """등록된 shared adapter family 메타데이터를 반환한다."""

    return _SHARED_ADAPTER_FAMILY_METADATA_REGISTRY

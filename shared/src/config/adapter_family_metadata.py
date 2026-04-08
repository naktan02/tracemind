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


__all__ = [
    "DIAGONAL_SCALE_FAMILY_METADATA",
    "SharedAdapterFamilyMetadata",
]

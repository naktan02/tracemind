"""Prototype payload serializer helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.domain.entities.artifacts.prototype_pack import (
    SingleCategoryPrototype,
    SinglePrototypePack,
)


@dataclass(slots=True)
class PrototypePackPayloadSpec:
    """PrototypePackPayload 공통 메타데이터 묶음."""

    schema_version: str
    prototype_version: str
    embedding_model_id: str
    embedding_model_revision: str
    translation_model_id: str | None
    translation_model_revision: str | None
    translation_direction: str | None
    mapping_version: str
    build_method: str
    distance_metric: str
    built_at: datetime


def build_prototype_pack_payload(
    *,
    spec: PrototypePackPayloadSpec,
    categories: Mapping[str, Sequence[Mapping[str, object]]],
) -> PrototypePackPayload:
    """공통 spec와 category payload로 PrototypePackPayload를 만든다."""
    return PrototypePackPayload.model_validate(
        {
            "schema_version": spec.schema_version,
            "prototype_version": spec.prototype_version,
            "embedding_model_id": spec.embedding_model_id,
            "embedding_model_revision": spec.embedding_model_revision,
            "translation_model_id": spec.translation_model_id,
            "translation_model_revision": spec.translation_model_revision,
            "translation_direction": spec.translation_direction,
            "mapping_version": spec.mapping_version,
            "build_method": spec.build_method,
            "distance_metric": spec.distance_metric,
            "built_at": spec.built_at,
            "categories": categories,
        }
    )


def build_single_prototype_categories(
    categories: Mapping[str, SingleCategoryPrototype],
) -> dict[str, list[dict[str, object]]]:
    """single-centroid category mapping을 canonical list shape로 바꾼다."""
    return {
        category: [
            {
                "prototype_id": f"{category}:single",
                "centroid": prototype.centroid,
                "sample_count": prototype.sample_count,
            }
        ]
        for category, prototype in categories.items()
    }


def build_single_prototype_pack_payload(
    pack: SinglePrototypePack,
) -> PrototypePackPayload:
    """single-centroid domain entity를 canonical payload로 직렬화한다."""
    return build_prototype_pack_payload(
        spec=PrototypePackPayloadSpec(
            schema_version=pack.schema_version,
            prototype_version=pack.prototype_version,
            embedding_model_id=pack.embedding_model_id,
            embedding_model_revision=pack.embedding_model_revision,
            translation_model_id=pack.translation_model_id,
            translation_model_revision=pack.translation_model_revision,
            translation_direction=pack.translation_direction,
            mapping_version=pack.mapping_version,
            build_method=pack.build_method,
            distance_metric=pack.distance_metric,
            built_at=pack.built_at,
        ),
        categories=build_single_prototype_categories(pack.categories),
    )


def describe_payload_spec(spec: PrototypePackPayloadSpec) -> dict[str, Any]:
    """manifest/debug용 payload spec dict."""
    return {
        "schema_version": spec.schema_version,
        "prototype_version": spec.prototype_version,
        "embedding_model_id": spec.embedding_model_id,
        "embedding_model_revision": spec.embedding_model_revision,
        "translation_model_id": spec.translation_model_id,
        "translation_model_revision": spec.translation_model_revision,
        "translation_direction": spec.translation_direction,
        "mapping_version": spec.mapping_version,
        "build_method": spec.build_method,
        "distance_metric": spec.distance_metric,
        "built_at": spec.built_at.isoformat(),
    }

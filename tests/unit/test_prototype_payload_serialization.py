"""Prototype payload serialization tests."""

from __future__ import annotations

from datetime import datetime, timezone

from shared.src.domain.entities.artifacts.prototype_pack import (
    SingleCategoryPrototype,
    SinglePrototypePack,
)
from shared.src.services.prototypes.payload_serialization import (
    PrototypePackPayloadSpec,
    build_prototype_pack_payload,
    build_single_prototype_pack_payload,
)


def test_build_single_prototype_pack_payload_uses_canonical_list_shape() -> None:
    pack = SinglePrototypePack(
        schema_version="prototype_pack.v1",
        prototype_version="proto_test_v1",
        embedding_model_id="hash_debug",
        embedding_model_revision="main",
        translation_model_id=None,
        translation_model_revision=None,
        translation_direction=None,
        mapping_version="ourafla_to_4cat.v1",
        build_method="mean_centroid_l2_normalized",
        distance_metric="cosine",
        built_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
        categories={
            "normal": SingleCategoryPrototype(
                centroid=[1.0, 0.0],
                sample_count=2,
            )
        },
    )

    payload = build_single_prototype_pack_payload(pack)

    assert list(payload.categories) == ["normal"]
    assert len(payload.categories["normal"]) == 1
    assert payload.categories["normal"][0].prototype_id == "normal:single"


def test_build_prototype_pack_payload_uses_shared_metadata_spec() -> None:
    payload = build_prototype_pack_payload(
        spec=PrototypePackPayloadSpec(
            schema_version="prototype_pack.v1",
            prototype_version="proto_test_v2",
            embedding_model_id="hash_debug",
            embedding_model_revision="main",
            translation_model_id=None,
            translation_model_revision=None,
            translation_direction=None,
            mapping_version="ourafla_to_4cat.v1",
            build_method="kmeans_mean_centroid_l2_normalized",
            distance_metric="cosine",
            built_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
        ),
        categories={
            "anxiety": [
                {
                    "prototype_id": "anxiety:kmeans:0",
                    "centroid": [1.0, 0.0],
                    "sample_count": 3,
                },
                {
                    "prototype_id": "anxiety:kmeans:1",
                    "centroid": [-1.0, 0.0],
                    "sample_count": 4,
                },
            ]
        },
    )

    assert payload.prototype_version == "proto_test_v2"
    assert payload.build_method == "kmeans_mean_centroid_l2_normalized"
    assert len(payload.categories["anxiety"]) == 2

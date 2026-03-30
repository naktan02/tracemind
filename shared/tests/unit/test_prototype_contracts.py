"""PrototypePack contract unit tests."""

from __future__ import annotations

import pytest

from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    extract_category_centroids,
    extract_category_prototypes,
)


def test_prototype_pack_payload_normalizes_legacy_single_centroid_shape() -> None:
    payload = PrototypePackPayload.model_validate(
        {
            "schema_version": "prototype_pack.v1",
            "prototype_version": "proto_test_v1",
            "embedding_model_id": "hash_debug",
            "embedding_model_revision": "v1",
            "mapping_version": "mapping.v1",
            "build_method": "mean_centroid_l2_normalized",
            "distance_metric": "cosine",
            "built_at": "2026-03-31T00:00:00Z",
            "categories": {
                "alert": {
                    "centroid": [1.0, 0.0],
                    "sample_count": 2,
                }
            },
        }
    )

    assert len(payload.categories["alert"]) == 1
    assert payload.categories["alert"][0].centroid == [1.0, 0.0]


def test_extract_category_prototypes_returns_all_vectors() -> None:
    payload = PrototypePackPayload.model_validate(
        {
            "schema_version": "prototype_pack.v1",
            "prototype_version": "proto_test_v2",
            "embedding_model_id": "hash_debug",
            "embedding_model_revision": "v1",
            "mapping_version": "mapping.v1",
            "build_method": "kmeans",
            "distance_metric": "cosine",
            "built_at": "2026-03-31T00:00:00Z",
            "categories": {
                "alert": [
                    {
                        "prototype_id": "alert:0",
                        "centroid": [1.0, 0.0],
                        "sample_count": 2,
                    },
                    {
                        "prototype_id": "alert:1",
                        "centroid": [0.0, 1.0],
                        "sample_count": 3,
                    },
                ]
            },
        }
    )

    assert extract_category_prototypes(payload)["alert"] == (
        [1.0, 0.0],
        [0.0, 1.0],
    )


def test_extract_category_centroids_rejects_multi_prototype_categories() -> None:
    payload = PrototypePackPayload.model_validate(
        {
            "schema_version": "prototype_pack.v1",
            "prototype_version": "proto_test_v2",
            "embedding_model_id": "hash_debug",
            "embedding_model_revision": "v1",
            "mapping_version": "mapping.v1",
            "build_method": "kmeans",
            "distance_metric": "cosine",
            "built_at": "2026-03-31T00:00:00Z",
            "categories": {
                "alert": [
                    {
                        "prototype_id": "alert:0",
                        "centroid": [1.0, 0.0],
                        "sample_count": 2,
                    },
                    {
                        "prototype_id": "alert:1",
                        "centroid": [0.0, 1.0],
                        "sample_count": 3,
                    },
                ]
            },
        }
    )

    with pytest.raises(ValueError, match="single-prototype categories"):
        extract_category_centroids(payload)

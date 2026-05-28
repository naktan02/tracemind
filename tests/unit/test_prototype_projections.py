"""Prototype projection helper tests."""

from __future__ import annotations

import pytest

from methods.prototype.projections import (
    project_category_centroids_by_largest_cluster,
    require_single_category_centroids,
)
from shared.src.contracts.prototype_contracts import PrototypePackPayload


def _single_payload() -> PrototypePackPayload:
    return PrototypePackPayload.model_validate(
        {
            "schema_version": "prototype_pack.v1",
            "prototype_version": "proto_single_v1",
            "embedding_model_id": "hash_debug",
            "embedding_model_revision": "main",
            "mapping_version": "ourafla_to_4cat.v1",
            "build_method": "mean_centroid_l2_normalized",
            "distance_metric": "cosine",
            "built_at": "2026-04-02T00:00:00+00:00",
            "categories": {
                "anxiety": [
                    {
                        "prototype_id": "anxiety:single",
                        "centroid": [1.0, 0.0],
                        "sample_count": 2,
                    }
                ]
            },
        }
    )


def _multi_payload() -> PrototypePackPayload:
    return PrototypePackPayload.model_validate(
        {
            "schema_version": "prototype_pack.v1",
            "prototype_version": "proto_multi_v1",
            "embedding_model_id": "hash_debug",
            "embedding_model_revision": "main",
            "mapping_version": "ourafla_to_4cat.v1",
            "build_method": "kmeans_mean_centroid_l2_normalized",
            "distance_metric": "cosine",
            "built_at": "2026-04-02T00:00:00+00:00",
            "categories": {
                "anxiety": [
                    {
                        "prototype_id": "anxiety:kmeans:0",
                        "centroid": [1.0, 0.0],
                        "sample_count": 2,
                    },
                    {
                        "prototype_id": "anxiety:kmeans:1",
                        "centroid": [0.0, 1.0],
                        "sample_count": 5,
                    },
                ]
            },
        }
    )


def test_require_single_category_centroids_accepts_single_payload() -> None:
    centroids = require_single_category_centroids(_single_payload())
    assert centroids == {"anxiety": [1.0, 0.0]}


def test_require_single_category_centroids_rejects_multi_payload() -> None:
    with pytest.raises(ValueError, match="single-prototype categories"):
        require_single_category_centroids(_multi_payload())


def test_project_category_centroids_by_largest_cluster_uses_representative() -> None:
    centroids = project_category_centroids_by_largest_cluster(_multi_payload())
    assert centroids == {"anxiety": [0.0, 1.0]}

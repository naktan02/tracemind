"""PrototypePackBuilder unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.prototype_pack_builder import PrototypePackBuilder


def test_build_creates_mean_centroids_for_each_category() -> None:
    builder = PrototypePackBuilder()

    pack = builder.build(
        {
            "anxiety": ([1.0, 3.0], [3.0, 5.0]),
            "normal": ([2.0, 2.0], [4.0, 4.0]),
        },
        prototype_version="proto_test_v1",
        embedding_model_id="hash_debug",
        embedding_model_revision="v1",
        translation_model_id=None,
        translation_model_revision=None,
        translation_direction=None,
        mapping_version="ourafla_to_4cat.v1",
        built_at=datetime(2026, 3, 28, tzinfo=timezone.utc),
    )

    assert pack.schema_version == "prototype_pack.v1"
    assert pack.build_method == "mean_centroid"
    assert pack.distance_metric == "cosine"
    assert pack.categories["anxiety"].centroid == pytest.approx([2.0, 4.0])
    assert pack.categories["anxiety"].sample_count == 2
    assert pack.categories["normal"].centroid == pytest.approx([3.0, 3.0])
    assert pack.categories["normal"].sample_count == 2


def test_build_rejects_missing_required_category() -> None:
    builder = PrototypePackBuilder()

    with pytest.raises(ValueError, match="Category 'suicidal' has no embeddings"):
        builder.build(
            {"anxiety": ([1.0, 2.0],)},
            prototype_version="proto_test_v1",
            embedding_model_id="hash_debug",
            embedding_model_revision="v1",
            translation_model_id=None,
            translation_model_revision=None,
            translation_direction=None,
            mapping_version="ourafla_to_4cat.v1",
            built_at=datetime(2026, 3, 28, tzinfo=timezone.utc),
            required_categories=("anxiety", "suicidal"),
        )


def test_build_rejects_mismatched_embedding_dimensions() -> None:
    builder = PrototypePackBuilder()

    with pytest.raises(ValueError, match="mismatched dimensions"):
        builder.build(
            {"anxiety": ([1.0, 2.0], [1.0, 2.0, 3.0])},
            prototype_version="proto_test_v1",
            embedding_model_id="hash_debug",
            embedding_model_revision="v1",
            translation_model_id=None,
            translation_model_revision=None,
            translation_direction=None,
            mapping_version="ourafla_to_4cat.v1",
            built_at=datetime(2026, 3, 28, tzinfo=timezone.utc),
        )

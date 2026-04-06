"""PrototypePackBuilder unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from shared.src.contracts.prototype_build_state_contracts import (
    SinglePrototypeBuildStatePayload,
)
from shared.src.services.prototypes.prototype_pack_builder import PrototypePackBuilder


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
    assert pack.build_method == "mean_centroid_l2_normalized"
    assert pack.distance_metric == "cosine"
    assert pack.categories["anxiety"].centroid == pytest.approx(
        [1.0 / 5.0**0.5, 2.0 / 5.0**0.5]
    )
    assert pack.categories["anxiety"].sample_count == 2
    assert pack.categories["normal"].centroid == pytest.approx([2.0**-0.5, 2.0**-0.5])
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


def test_build_rejects_zero_norm_mean_centroid() -> None:
    builder = PrototypePackBuilder()

    with pytest.raises(ValueError, match="zero norm"):
        builder.build(
            {"anxiety": ([1.0, 0.0], [-1.0, 0.0])},
            prototype_version="proto_test_v1",
            embedding_model_id="hash_debug",
            embedding_model_revision="v1",
            translation_model_id=None,
            translation_model_revision=None,
            translation_direction=None,
            mapping_version="ourafla_to_4cat.v1",
            built_at=datetime(2026, 3, 28, tzinfo=timezone.utc),
        )


def test_build_state_and_pack_from_state_preserve_exact_mean() -> None:
    builder = PrototypePackBuilder()

    build_state = builder.build_state(
        {
            "anxiety": ([1.0, 3.0], [3.0, 5.0]),
            "normal": ([2.0, 2.0], [4.0, 4.0]),
        },
        prototype_version="proto_test_v1",
        embedding_backend="hash_debug",
        embedding_model_id="hash_debug",
        embedding_model_revision="v1",
        normalize_embeddings=True,
        task_prefix="",
        translation_model_id=None,
        translation_model_revision=None,
        translation_direction=None,
        mapping_version="ourafla_to_4cat.v1",
        built_at=datetime(2026, 3, 28, tzinfo=timezone.utc),
    )
    pack = builder.build_pack_from_state(build_state)

    assert build_state.schema_version == "prototype_build_state.v1"
    assert build_state.embedding_backend == "hash_debug"
    assert build_state.categories["anxiety"].embedding_sum == pytest.approx([4.0, 8.0])
    assert build_state.categories["anxiety"].sample_count == 2
    assert pack.categories["anxiety"].centroid == pytest.approx(
        [1.0 / 5.0**0.5, 2.0 / 5.0**0.5]
    )


def test_merge_build_state_accumulates_new_embeddings_exactly() -> None:
    builder = PrototypePackBuilder()
    base_state = SinglePrototypeBuildStatePayload(
        schema_version="prototype_build_state.v1",
        prototype_version="proto_base_v1",
        embedding_backend="hash_debug",
        embedding_model_id="hash_debug",
        embedding_model_revision="v1",
        normalize_embeddings=True,
        task_prefix="",
        translation_model_id=None,
        translation_model_revision=None,
        translation_direction=None,
        mapping_version="ourafla_to_4cat.v1",
        build_method="mean_centroid_l2_normalized",
        distance_metric="cosine",
        built_at=datetime(2026, 3, 28, tzinfo=timezone.utc),
        categories={
            "anxiety": {
                "embedding_sum": [4.0, 8.0],
                "sample_count": 2,
            },
            "normal": {
                "embedding_sum": [6.0, 6.0],
                "sample_count": 2,
            },
        },
    )

    merged_state = builder.merge_build_state(
        base_state,
        {
            "anxiety": ([5.0, 7.0],),
            "normal": ([6.0, 4.0], [8.0, 6.0]),
        },
        prototype_version="proto_next_v1",
        built_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
    )
    pack = builder.build_pack_from_state(merged_state)

    assert merged_state.prototype_version == "proto_next_v1"
    assert merged_state.categories["anxiety"].embedding_sum == pytest.approx(
        [9.0, 15.0]
    )
    assert merged_state.categories["anxiety"].sample_count == 3
    assert merged_state.categories["normal"].embedding_sum == pytest.approx(
        [20.0, 16.0]
    )
    assert merged_state.categories["normal"].sample_count == 4
    assert pack.categories["anxiety"].centroid == pytest.approx(
        [3.0 / 34.0**0.5, 5.0 / 34.0**0.5]
    )


def test_merge_build_state_rejects_unexpected_new_category() -> None:
    builder = PrototypePackBuilder()
    base_state = SinglePrototypeBuildStatePayload(
        schema_version="prototype_build_state.v1",
        prototype_version="proto_base_v1",
        embedding_backend="hash_debug",
        embedding_model_id="hash_debug",
        embedding_model_revision="v1",
        normalize_embeddings=True,
        task_prefix="",
        translation_model_id=None,
        translation_model_revision=None,
        translation_direction=None,
        mapping_version="ourafla_to_4cat.v1",
        build_method="mean_centroid_l2_normalized",
        distance_metric="cosine",
        built_at=datetime(2026, 3, 28, tzinfo=timezone.utc),
        categories={
            "anxiety": {
                "embedding_sum": [4.0, 8.0],
                "sample_count": 2,
            },
        },
    )

    with pytest.raises(ValueError, match="Unexpected categories"):
        builder.merge_build_state(
            base_state,
            {"normal": ([1.0, 1.0],)},
            prototype_version="proto_next_v1",
            built_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        )

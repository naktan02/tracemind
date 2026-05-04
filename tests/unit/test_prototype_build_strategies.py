"""Production prototype build strategy tests."""

from __future__ import annotations

from datetime import datetime, timezone

from shared.src.services.prototypes.build_strategies import (
    DbscanPrototypeBuildStrategy,
    KMeansPrototypeBuildStrategy,
    PrototypeBuildRequest,
    SinglePrototypeBuildStrategy,
)


def _request(
    embeddings_by_category: dict[str, tuple[tuple[float, ...], ...]],
) -> PrototypeBuildRequest:
    return PrototypeBuildRequest(
        embeddings_by_category=embeddings_by_category,
        prototype_version="proto_test_v1",
        embedding_backend="hash_debug",
        embedding_model_id="hash_debug",
        embedding_model_revision="test",
        mapping_version="ourafla_to_4cat.v1",
        built_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
    )


def test_single_strategy_builds_one_prototype_and_exact_build_state() -> None:
    strategy = SinglePrototypeBuildStrategy()

    result = strategy.build(
        _request(
            {
                "anxiety": ((1.0, 3.0), (3.0, 5.0)),
                "normal": ((2.0, 2.0), (4.0, 4.0)),
            }
        )
    )

    assert result.build_state_payload is not None
    assert result.pack_payload.build_method == "mean_centroid_l2_normalized"
    assert len(result.pack_payload.categories["anxiety"]) == 1
    assert result.pack_payload.categories["anxiety"][0].prototype_id == "anxiety:single"


def test_kmeans_strategy_builds_multi_prototypes_without_build_state() -> None:
    strategy = KMeansPrototypeBuildStrategy(
        candidate_ks=(2,),
        silhouette_sample_size=10,
        random_state=0,
    )

    result = strategy.build(
        _request(
            {
                "anxiety": (
                    (1.0, 0.0),
                    (1.0, 0.1),
                    (-1.0, 0.0),
                    (-1.0, -0.1),
                ),
                "normal": ((0.0, 1.0), (0.0, 1.1)),
            }
        )
    )

    assert result.build_state_payload is None
    assert result.pack_payload.build_method == "kmeans_mean_centroid_l2_normalized"
    assert len(result.pack_payload.categories["anxiety"]) == 2
    assert {
        prototype.prototype_id
        for prototype in result.pack_payload.categories["anxiety"]
    } == {"anxiety:kmeans:0", "anxiety:kmeans:1"}


def test_dbscan_strategy_builds_multi_prototypes_and_metadata() -> None:
    strategy = DbscanPrototypeBuildStrategy(
        eps_values=(0.05, 0.1, 0.2),
        min_samples_values=(2,),
        search_sample_size=10,
        min_cluster_coverage=0.9,
        random_state=0,
    )

    result = strategy.build(
        _request(
            {
                "anxiety": (
                    (1.0, 0.0),
                    (0.99, 0.03),
                    (0.96, -0.08),
                    (0.0, 1.0),
                    (0.03, 0.99),
                    (-0.08, 0.96),
                )
            }
        )
    )

    assert result.build_state_payload is None
    assert result.pack_payload.build_method == "dbscan_mean_centroid_l2_normalized"
    assert len(result.pack_payload.categories["anxiety"]) >= 2
    assert result.metadata["labels"]["anxiety"]["fallback"] is False
    assert result.metadata["labels"]["anxiety"]["member_counts"] == [3, 3]


def test_dbscan_strategy_falls_back_for_too_few_examples() -> None:
    strategy = DbscanPrototypeBuildStrategy(
        eps_values=(0.05,),
        min_samples_values=(2,),
    )

    result = strategy.build(
        _request(
            {
                "normal": ((1.0, 0.0),),
            }
        )
    )

    assert len(result.pack_payload.categories["normal"]) == 1
    assert result.pack_payload.categories["normal"][0].prototype_id == "normal:single"
    assert result.metadata["labels"]["normal"] == {
        "fallback": True,
        "reason": "too_few_examples",
    }

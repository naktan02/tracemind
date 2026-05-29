from __future__ import annotations

import pytest

from methods.prototype.distance_report import (
    build_pairwise_distance_report,
    render_pairwise_table,
    resolve_prototype_centroid_view,
)
from methods.prototype.evaluation import evaluate_prototype_pack_rows
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.prototype_contracts import PrototypePackPayload


def _row(query_id: str, label: str, text: str) -> LabeledQueryRow:
    return LabeledQueryRow(
        query_id=query_id,
        text=text,
        raw_label_scheme="manual_label",
        raw_label=label,
        mapped_label_4=label,
        locale="en-US",
        annotation_source="unit_test",
        approved_by="tester",
        created_at="2026-05-28T00:00:00+00:00",
    )


def test_evaluate_prototype_pack_rows_uses_prototype_similarity_scores() -> None:
    report = evaluate_prototype_pack_rows(
        rows=[
            _row("q1", "anxiety", "anxious text"),
            _row("q2", "normal", "calm text"),
        ],
        prototypes={
            "anxiety": ([1.0, 0.0],),
            "normal": ([0.0, 1.0],),
        },
        embeddings=[
            [0.9, 0.1],
            [0.1, 0.9],
        ],
    )

    assert report["rows_total"] == 2
    assert report["accuracy_top_1"] == 1.0
    assert report["correct_top_1"] == 2
    assert report["per_category"]["anxiety"]["support"] == 1
    assert report["per_category"]["normal"]["support"] == 1


def test_pairwise_distance_report_uses_resolved_centroid_view() -> None:
    payload = PrototypePackPayload.model_validate(
        {
            "schema_version": "prototype_pack.v1",
            "prototype_version": "prototype_pack.v1",
            "embedding_model_id": "hash",
            "embedding_model_revision": "test",
            "mapping_version": "unit_test",
            "build_method": "kmeans",
            "distance_metric": "cosine",
            "built_at": "2026-05-28T00:00:00+00:00",
            "categories": {
                "anxiety": [
                    {
                        "prototype_id": "anxiety:small",
                        "centroid": [0.5, 0.5],
                        "sample_count": 1,
                    },
                    {
                        "prototype_id": "anxiety:large",
                        "centroid": [1.0, 0.0],
                        "sample_count": 3,
                    },
                ],
                "normal": [
                    {
                        "prototype_id": "normal:only",
                        "centroid": [0.0, 1.0],
                        "sample_count": 2,
                    }
                ],
            },
        }
    )

    centroids = resolve_prototype_centroid_view(
        payload=payload,
        centroid_view="largest_cluster",
    )
    report = build_pairwise_distance_report(centroids)
    table = render_pairwise_table(
        title="l2_distance",
        categories=report.categories,
        values=report.l2_values,
    )

    assert centroids["anxiety"] == [1.0, 0.0]
    assert report.cosine_values[("anxiety", "normal")] == pytest.approx(0.0)
    assert report.l2_values[("anxiety", "normal")] == pytest.approx(2**0.5)
    assert "| category | anxiety | normal |" in table

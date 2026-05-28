from __future__ import annotations

from methods.prototype.evaluation import evaluate_prototype_pack_rows
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


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

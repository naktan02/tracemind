"""labeled query row contract helper tests."""

from __future__ import annotations

from shared.src.contracts.labeled_query_row_contracts import (
    count_labeled_query_rows_by_label,
    get_labeled_query_row_mapped_label,
    group_labeled_query_rows_by_label,
)


def _row(query_id: str, label: str) -> dict[str, str | None]:
    return {
        "query_id": query_id,
        "text": f"{label} text",
        "raw_label_scheme": "mapped_label_4",
        "raw_label": label,
        "mapped_label_4": label,
        "locale": "en",
        "annotation_source": "test",
        "approved_by": None,
        "created_at": "2026-01-01T00:00:00Z",
    }


def test_labeled_query_row_mapped_label_helper_is_canonical() -> None:
    assert get_labeled_query_row_mapped_label(_row("q1", "anxiety")) == "anxiety"


def test_labeled_query_rows_group_and_count_by_canonical_label() -> None:
    rows = [
        _row("q1", "normal"),
        _row("q2", "anxiety"),
        _row("q3", "normal"),
    ]

    assert group_labeled_query_rows_by_label(rows) == {
        "anxiety": [rows[1]],
        "normal": [rows[0], rows[2]],
    }
    assert count_labeled_query_rows_by_label(rows) == {
        "anxiety": 1,
        "normal": 2,
    }

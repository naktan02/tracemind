"""Query adaptation data preparation tests."""

from __future__ import annotations

from methods.adaptation.query_classifier_adaptation.data import (
    TextMultiviewDataset,
    TextWeakDataset,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


def _row(query_id: str, text: str) -> LabeledQueryRow:
    return LabeledQueryRow(
        query_id=query_id,
        text=text,
        raw_label_scheme="manual_label",
        raw_label="anxiety",
        mapped_label_4="anxiety",
        locale="en-US",
        annotation_source="unit_test",
        approved_by="annotator",
        created_at="2026-04-19T00:00:00+00:00",
    )


def test_text_multiview_dataset_uses_usb_aug_candidates(monkeypatch) -> None:
    row = _row("q1", "I feel anxious today.")
    row["aug_0"] = "I feel nervous today."
    row["aug_1"] = "Today I feel uneasy."

    monkeypatch.setattr(
        "methods.adaptation.query_classifier_adaptation.data.random.choice",
        lambda choices: choices[1],
    )

    dataset = TextMultiviewDataset(rows=[row], task_prefix="label: ")
    item = dataset[0]

    assert item["query_id"] == "q1"
    assert item["weak_text"] == "label: I feel anxious today."
    assert item["strong_text"] == "label: Today I feel uneasy."


def test_text_multiview_dataset_keeps_legacy_weak_strong_compatibility() -> None:
    row = _row("q2", "I feel low.")
    row["weak_text"] = "weak::I feel low."
    row["strong_text"] = "strong::I feel low."

    dataset = TextMultiviewDataset(rows=[row], task_prefix="")
    item = dataset[0]

    assert item["weak_text"] == "weak::I feel low."
    assert item["strong_text"] == "strong::I feel low."


def test_text_weak_dataset_uses_original_text_as_usb_weak_view() -> None:
    row = _row("q3", "I feel anxious today.")

    dataset = TextWeakDataset(rows=[row], task_prefix="label: ")
    item = dataset[0]

    assert item["query_id"] == "q3"
    assert item["weak_text"] == "label: I feel anxious today."


def test_text_weak_dataset_keeps_legacy_weak_text_compatibility() -> None:
    row = _row("q4", "I feel low.")
    row["weak_text"] = "weak::I feel low."

    dataset = TextWeakDataset(rows=[row], task_prefix="")
    item = dataset[0]

    assert item["weak_text"] == "weak::I feel low."

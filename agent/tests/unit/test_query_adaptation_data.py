"""Query adaptation data preparation tests."""

from __future__ import annotations

from agent.src.services.training.query_adaptation.data import TextMultiviewDataset
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
        "agent.src.services.training.query_adaptation.data.random.choice",
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

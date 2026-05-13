"""Query adaptation data preparation tests."""

from __future__ import annotations

from methods.adaptation.query_classifier_adaptation.data import (
    TextMultiviewDataset,
    TextWeakDataset,
    build_multiview_dataloader,
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


def test_text_multiview_dataset_uses_first_usb_aug_candidate() -> None:
    row = _row("q1", "I feel anxious today.")
    row["aug_0"] = "I feel nervous today."
    row["aug_1"] = "Today I feel uneasy."

    dataset = TextMultiviewDataset(rows=[row], task_prefix="label: ")
    item = dataset[0]

    assert item["query_id"] == "q1"
    assert item["row_index"] == 0
    assert item["weak_text"] == "label: I feel anxious today."
    assert item["strong_text"] == "label: I feel nervous today."


def test_text_multiview_dataset_can_use_second_usb_aug_candidate() -> None:
    row = _row("q1", "I feel anxious today.")
    row["aug_0"] = "I feel nervous today."
    row["aug_1"] = "Today I feel uneasy."

    dataset = TextMultiviewDataset(
        rows=[row],
        task_prefix="",
        strong_view_policy="second_aug",
    )
    item = dataset[0]

    assert item["strong_text"] == "Today I feel uneasy."


def test_text_multiview_dataset_can_alternate_usb_aug_candidate_by_row() -> None:
    rows = [_row("q1", "first"), _row("q2", "second")]
    for row in rows:
        row["aug_0"] = f"de::{row['text']}"
        row["aug_1"] = f"fr::{row['text']}"

    dataset = TextMultiviewDataset(
        rows=rows,
        task_prefix="",
        strong_view_policy="row_parity_aug",
    )

    assert dataset[0]["strong_text"] == "de::first"
    assert dataset[1]["strong_text"] == "fr::second"


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
    assert item["row_index"] == 0
    assert item["weak_text"] == "label: I feel anxious today."


def test_text_weak_dataset_keeps_legacy_weak_text_compatibility() -> None:
    row = _row("q4", "I feel low.")
    row["weak_text"] = "weak::I feel low."

    dataset = TextWeakDataset(rows=[row], task_prefix="")
    item = dataset[0]

    assert item["weak_text"] == "weak::I feel low."


def test_multiview_dataloader_emits_stable_row_indices() -> None:
    class _Tokenizer:
        def __call__(self, texts, **_kwargs):
            import torch

            return {
                "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
            }

    rows = [_row("q1", "first"), _row("q2", "second")]
    for row in rows:
        row["aug_0"] = f"de::{row['text']}"
        row["aug_1"] = f"fr::{row['text']}"

    loader = build_multiview_dataloader(
        rows=rows,
        tokenizer=_Tokenizer(),
        batch_size=2,
        max_length=8,
        task_prefix="",
        shuffle=False,
    )
    batch = next(iter(loader))

    assert batch["query_ids"] == ["q1", "q2"]
    assert batch["row_indices"].tolist() == [0, 1]

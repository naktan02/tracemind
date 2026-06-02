"""Query adaptation data preparation tests."""

from __future__ import annotations

from methods.adaptation.query_text_views.data import (
    TextLabelDataset,
    TextMultiviewDataset,
    TextWeakDataset,
    TextWeakStrongPairDataset,
    build_dataloader,
    build_multiview_dataloader,
    build_weak_dataloader,
    build_weak_strong_pair_dataloader,
)
from methods.adaptation.query_text_views.query_ssl_views import (
    build_query_ssl_unlabeled_dataloader,
)
from methods.adaptation.query_text_views.tokenization import (
    TextTokenizationCache,
)
from methods.adaptation.query_text_views.view_rows import (
    row_supports_query_ssl_view_builder,
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


def test_text_label_dataset_exposes_label_histogram() -> None:
    rows = [
        _row("q1", "I feel anxious."),
        _row("q2", "I feel low."),
        _row("q3", "I feel better."),
    ]
    rows[1]["mapped_label_4"] = "depression"
    rows[2]["mapped_label_4"] = "anxiety"

    dataset = TextLabelDataset(
        rows=rows,
        label_to_index={"anxiety": 0, "depression": 1},
        task_prefix="",
    )

    assert dataset.label_histogram(num_classes=2).tolist() == [2.0, 1.0]


def test_text_multiview_dataset_keeps_legacy_weak_strong_compatibility() -> None:
    row = _row("q2", "I feel low.")
    row["weak_text"] = "weak::I feel low."
    row["strong_text"] = "strong::I feel low."

    dataset = TextMultiviewDataset(rows=[row], task_prefix="")
    item = dataset[0]

    assert item["weak_text"] == "weak::I feel low."
    assert item["strong_text"] == "strong::I feel low."


def test_text_weak_strong_pair_dataset_exposes_both_usb_aug_candidates() -> None:
    row = _row("q1", "I feel anxious today.")
    row["aug_0"] = "I feel nervous today."
    row["aug_1"] = "Today I feel uneasy."

    dataset = TextWeakStrongPairDataset(rows=[row], task_prefix="label: ")
    item = dataset[0]

    assert item["query_id"] == "q1"
    assert item["row_index"] == 0
    assert item["weak_text"] == "label: I feel anxious today."
    assert item["strong_0_text"] == "label: I feel nervous today."
    assert item["strong_1_text"] == "label: Today I feel uneasy."


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


def test_labeled_dataloader_emits_stable_row_indices() -> None:
    class _Tokenizer:
        def __call__(self, texts, **_kwargs):
            import torch

            return {
                "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
            }

    rows = [_row("q1", "first"), _row("q2", "second")]

    loader = build_dataloader(
        rows=rows,
        label_to_index={"anxiety": 0},
        tokenizer=_Tokenizer(),
        batch_size=2,
        max_length=8,
        task_prefix="",
        shuffle=False,
    )
    batch = next(iter(loader))

    assert batch["row_indices"].tolist() == [0, 1]


def test_weak_strong_pair_dataloader_emits_both_strong_views() -> None:
    class _Tokenizer:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, texts, **_kwargs):
            import torch

            self.calls.append(list(texts))
            return {
                "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
            }

    row = _row("q1", "base")
    row["aug_0"] = "first strong"
    row["aug_1"] = "second strong"
    tokenizer = _Tokenizer()

    loader = build_weak_strong_pair_dataloader(
        rows=[row],
        tokenizer=tokenizer,
        batch_size=1,
        max_length=8,
        task_prefix="prompt: ",
        shuffle=False,
    )
    batch = next(iter(loader))

    assert tokenizer.calls == [
        ["prompt: base"],
        ["prompt: first strong"],
        ["prompt: second strong"],
    ]
    assert batch["query_ids"] == ["q1"]
    assert batch["row_indices"].tolist() == [0]
    assert "strong_input_ids" not in batch
    assert "strong_0_input_ids" in batch
    assert "strong_1_input_ids" in batch


def test_query_ssl_unlabeled_loader_dispatches_multiview_surface() -> None:
    class _Tokenizer:
        def __call__(self, texts, **_kwargs):
            import torch

            return {
                "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
            }

    row = _row("q1", "base")
    row["aug_0"] = "first strong"
    row["aug_1"] = "second strong"

    loader = build_query_ssl_unlabeled_dataloader(
        rows=[row],
        tokenizer=_Tokenizer(),
        batch_size=1,
        max_length=8,
        task_prefix="",
        shuffle=False,
        view_builder_name="usb_multiview",
        strong_view_policy="second_aug",
    )
    batch = next(iter(loader))

    assert batch["query_ids"] == ["q1"]
    assert "strong_input_ids" in batch


def test_query_ssl_unlabeled_loader_dispatches_weak_strong_pair_surface() -> None:
    class _Tokenizer:
        def __call__(self, texts, **_kwargs):
            import torch

            return {
                "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
            }

    row = _row("q1", "base")
    row["aug_0"] = "first strong"
    row["aug_1"] = "second strong"

    loader = build_query_ssl_unlabeled_dataloader(
        rows=[row],
        tokenizer=_Tokenizer(),
        batch_size=1,
        max_length=8,
        task_prefix="",
        shuffle=False,
        view_builder_name="usb_weak_strong_pair",
    )
    batch = next(iter(loader))

    assert batch["query_ids"] == ["q1"]
    assert "strong_0_input_ids" in batch
    assert "strong_1_input_ids" in batch
    assert "strong_input_ids" not in batch


def test_query_ssl_unlabeled_loader_dispatches_weak_surface() -> None:
    class _Tokenizer:
        def __call__(self, texts, **_kwargs):
            import torch

            return {
                "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
            }

    loader = build_query_ssl_unlabeled_dataloader(
        rows=[_row("q1", "base")],
        tokenizer=_Tokenizer(),
        batch_size=1,
        max_length=8,
        task_prefix="",
        shuffle=False,
        view_builder_name="usb_weak",
    )
    batch = next(iter(loader))

    assert batch["query_ids"] == ["q1"]
    assert "weak_input_ids" in batch
    assert "strong_input_ids" not in batch


def test_weak_strong_pair_view_builder_requires_strict_usb_candidates() -> None:
    strict_row = _row("q1", "base")
    strict_row["aug_0"] = "first strong"
    strict_row["aug_1"] = "second strong"
    legacy_row = _row("q2", "base")
    legacy_row["weak_text"] = "weak"
    legacy_row["strong_text"] = "strong"

    assert row_supports_query_ssl_view_builder(
        row=strict_row,
        view_builder_name="usb_weak_strong_pair",
    )
    assert not row_supports_query_ssl_view_builder(
        row=legacy_row,
        view_builder_name="usb_weak_strong_pair",
    )


def test_text_tokenization_cache_reuses_selected_texts() -> None:
    class _Tokenizer:
        pad_token_id = 0
        padding_side = "right"
        name_or_path = "unit-tokenizer"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def __call__(self, texts, **_kwargs):
            self.calls.append(str(texts))
            values = [ord(char) % 17 + 1 for char in str(texts)]
            return {
                "input_ids": values,
                "attention_mask": [1 for _value in values],
            }

    row = _row("q1", "same text")
    cache = TextTokenizationCache()
    tokenizer = _Tokenizer()

    first_loader = build_weak_dataloader(
        rows=[row],
        tokenizer=tokenizer,
        batch_size=1,
        max_length=8,
        task_prefix="",
        shuffle=False,
        tokenization_cache=cache,
        tokenization_cache_namespace="unit",
    )
    second_loader = build_weak_dataloader(
        rows=[row],
        tokenizer=tokenizer,
        batch_size=1,
        max_length=8,
        task_prefix="",
        shuffle=False,
        tokenization_cache=cache,
        tokenization_cache_namespace="unit",
    )

    first_batch = next(iter(first_loader))
    second_batch = next(iter(second_loader))

    assert tokenizer.calls == ["same text"]
    assert (
        first_batch["weak_input_ids"].tolist()
        == second_batch["weak_input_ids"].tolist()
    )

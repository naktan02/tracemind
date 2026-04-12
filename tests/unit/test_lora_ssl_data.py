from __future__ import annotations

import torch
import pytest

from scripts.experiments.lora_classifier.ssl_data import (
    SemiSupervisedTextBatch,
    build_semi_supervised_text_batch_loader,
)
from scripts.experiments.lora_classifier.data import build_label_index


class _Tokenizer:
    def __call__(
        self,
        texts,
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ):
        del padding, truncation, return_tensors
        max_tokens = max(min(len(text.split()), max_length) for text in texts)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for text in texts:
            token_count = min(len(text.split()), max_length)
            row = list(range(1, token_count + 1))
            padded = row + [0] * (max_tokens - token_count)
            mask = [1] * token_count + [0] * (max_tokens - token_count)
            input_ids.append(padded)
            attention_mask.append(mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def test_build_semi_supervised_text_batch_loader_builds_three_branch_batches() -> None:
    labeled_rows = [
        _build_labeled_row("l1", "불안해서 잠이 안 와요", "anxiety"),
        _build_labeled_row("l2", "요즘 우울하고 의욕이 없어요", "depression"),
        _build_labeled_row("l3", "오늘은 평소보다 괜찮아요", "normal"),
    ]
    unlabeled_rows = [
        _build_unlabeled_row("u1", "원문 하나", "약하게 하나", "강하게 하나"),
        _build_unlabeled_row("u2", "원문 둘", "약하게 둘", "강하게 둘"),
        _build_unlabeled_row("u3", "원문 셋", "약하게 셋", "강하게 셋"),
        _build_unlabeled_row("u4", "원문 넷", "약하게 넷", "강하게 넷"),
    ]
    _, label_to_index = build_label_index(labeled_rows)

    loader = build_semi_supervised_text_batch_loader(
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        label_to_index=label_to_index,
        tokenizer=_Tokenizer(),
        labeled_batch_size=2,
        unlabeled_batch_size_multiplier=2,
        max_length=8,
        task_prefix="task: ",
        shuffle=False,
        seed=7,
    )

    batches = list(loader)

    assert len(loader) == 2
    assert len(batches) == 2
    first_batch = batches[0]
    assert isinstance(first_batch, SemiSupervisedTextBatch)
    assert first_batch.labeled_query_ids == ("l1", "l2")
    assert first_batch.unlabeled_query_ids == ("u1", "u2", "u3", "u4")
    assert tuple(first_batch.labeled_labels.tolist()) == (0, 1)
    assert first_batch.labeled_input_ids.shape[0] == 2
    assert first_batch.weak_unlabeled_input_ids.shape[0] == 4
    assert first_batch.strong_unlabeled_input_ids.shape[0] == 4

    second_batch = batches[1]
    assert second_batch.labeled_query_ids == ("l3",)
    assert second_batch.unlabeled_query_ids == ("u1", "u2", "u3", "u4")


def test_build_semi_supervised_text_batch_loader_rejects_missing_multiview_fields() -> None:
    labeled_rows = [_build_labeled_row("l1", "불안해서 잠이 안 와요", "anxiety")]
    _, label_to_index = build_label_index(labeled_rows)

    with pytest.raises(ValueError, match="strong_text must be a non-empty string."):
        list(
            build_semi_supervised_text_batch_loader(
                labeled_rows=labeled_rows,
                unlabeled_rows=[
                    {
                        "query_id": "u1",
                        "weak_text": "약한 view",
                    }
                ],
                label_to_index=label_to_index,
                tokenizer=_Tokenizer(),
                labeled_batch_size=1,
                unlabeled_batch_size_multiplier=1,
                max_length=8,
                task_prefix="",
                shuffle=False,
            )
        )


def _build_labeled_row(query_id: str, text: str, label: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "text": text,
        "raw_label_scheme": "gold",
        "raw_label": label,
        "mapped_label_4": label,
        "locale": "ko-KR",
        "annotation_source": "test",
        "approved_by": "tester",
        "created_at": "2026-04-13T00:00:00+00:00",
    }


def _build_unlabeled_row(
    query_id: str,
    text: str,
    weak_text: str,
    strong_text: str,
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "text": text,
        "weak_text": weak_text,
        "strong_text": strong_text,
    }

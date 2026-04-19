"""Query adaptation supervised / FixMatch baseline용 데이터 준비 유틸리티."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from scripts.labeled_query_rows import LabeledQueryRow


class TextLabelDataset(Dataset[dict[str, Any]]):
    """라벨된 query row를 tokenizer 입력 배치로 노출한다."""

    def __init__(
        self,
        *,
        rows: list[LabeledQueryRow],
        label_to_index: dict[str, int],
        task_prefix: str,
    ) -> None:
        self._rows = rows
        self._label_to_index = label_to_index
        self._task_prefix = task_prefix

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        text = str(row["text"])
        if self._task_prefix:
            text = f"{self._task_prefix}{text}"
        return {
            "text": text,
            "label": self._label_to_index[str(row["mapped_label_4"])],
        }


class TextMultiviewDataset(Dataset[dict[str, Any]]):
    """weak/strong text view가 있는 unlabeled row를 배치용으로 노출한다."""

    def __init__(
        self,
        *,
        rows: list[LabeledQueryRow],
        task_prefix: str,
    ) -> None:
        self._rows = rows
        self._task_prefix = task_prefix

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        weak_text = row.get("weak_text")
        strong_text = row.get("strong_text")
        if weak_text is None or strong_text is None:
            raise ValueError(
                "FixMatch unlabeled rows require both weak_text and strong_text."
            )
        weak_text = str(weak_text)
        strong_text = str(strong_text)
        if self._task_prefix:
            weak_text = f"{self._task_prefix}{weak_text}"
            strong_text = f"{self._task_prefix}{strong_text}"
        return {
            "query_id": str(row["query_id"]),
            "weak_text": weak_text,
            "strong_text": strong_text,
        }


def build_label_index(
    rows: list[LabeledQueryRow],
) -> tuple[list[str], dict[str, int]]:
    """학습 row에서 canonical label index를 만든다."""

    categories = sorted({str(row["mapped_label_4"]) for row in rows})
    return categories, {category: index for index, category in enumerate(categories)}


def build_dataloader(
    *,
    rows: list[LabeledQueryRow],
    label_to_index: dict[str, int],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    task_prefix: str,
    shuffle: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Labeled row를 baseline trainer 입력 DataLoader로 변환한다."""

    dataset = TextLabelDataset(
        rows=rows,
        label_to_index=label_to_index,
        task_prefix=task_prefix,
    )

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [str(item["text"]) for item in batch]
        labels = [int(item["label"]) for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )


def build_multiview_dataloader(
    *,
    rows: list[LabeledQueryRow],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    task_prefix: str,
    shuffle: bool,
) -> DataLoader[dict[str, Any]]:
    """weak/strong unlabeled row를 FixMatch 입력 DataLoader로 변환한다."""

    dataset = TextMultiviewDataset(
        rows=rows,
        task_prefix=task_prefix,
    )

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        weak_texts = [str(item["weak_text"]) for item in batch]
        strong_texts = [str(item["strong_text"]) for item in batch]
        weak_encoded = tokenizer(
            weak_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        strong_encoded = tokenizer(
            strong_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "query_ids": [str(item["query_id"]) for item in batch],
            "weak_input_ids": weak_encoded["input_ids"],
            "weak_attention_mask": weak_encoded["attention_mask"],
            "strong_input_ids": strong_encoded["input_ids"],
            "strong_attention_mask": strong_encoded["attention_mask"],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )


__all__ = [
    "TextMultiviewDataset",
    "TextLabelDataset",
    "build_dataloader",
    "build_label_index",
    "build_multiview_dataloader",
]

"""LoRA classifier 실험용 데이터 준비 유틸리티."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from scripts.labeled_query_rows import LabeledQueryRow


class TextLabelDataset(Dataset[dict[str, Any]]):
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


def build_label_index(
    rows: list[LabeledQueryRow],
) -> tuple[list[str], dict[str, int]]:
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

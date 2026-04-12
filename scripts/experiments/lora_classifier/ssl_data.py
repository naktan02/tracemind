"""LoRA SSL 실험용 3-branch batch 준비 유틸리티."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from scripts.labeled_query_rows import LabeledQueryRow


@dataclass(slots=True)
class SemiSupervisedTextBatch:
    """labeled + weak unlabeled + strong unlabeled를 묶은 batch."""

    labeled_query_ids: tuple[str, ...]
    unlabeled_query_ids: tuple[str, ...]
    labeled_input_ids: torch.Tensor
    labeled_attention_mask: torch.Tensor
    labeled_labels: torch.Tensor
    weak_unlabeled_input_ids: torch.Tensor
    weak_unlabeled_attention_mask: torch.Tensor
    strong_unlabeled_input_ids: torch.Tensor
    strong_unlabeled_attention_mask: torch.Tensor


class _LabeledTextDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        rows: Sequence[LabeledQueryRow],
        label_to_index: Mapping[str, int],
        task_prefix: str,
    ) -> None:
        self._rows = list(rows)
        self._label_to_index = dict(label_to_index)
        self._task_prefix = task_prefix

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        text = str(row["text"])
        if self._task_prefix:
            text = f"{self._task_prefix}{text}"
        return {
            "query_id": str(row["query_id"]),
            "text": text,
            "label": int(self._label_to_index[str(row["mapped_label_4"])]),
        }


class _UnlabeledMultiviewTextDataset(Dataset[dict[str, str]]):
    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, object]],
        task_prefix: str,
    ) -> None:
        self._rows = list(rows)
        self._task_prefix = task_prefix

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self._rows[index]
        weak_text = _require_row_str(row, "weak_text")
        strong_text = _require_row_str(row, "strong_text")
        if self._task_prefix:
            weak_text = f"{self._task_prefix}{weak_text}"
            strong_text = f"{self._task_prefix}{strong_text}"
        return {
            "query_id": _require_row_str(row, "query_id"),
            "weak_text": weak_text,
            "strong_text": strong_text,
        }


@dataclass(slots=True)
class SemiSupervisedTextBatchLoader:
    """공식 FixMatch와 같은 3-branch 입력을 만드는 로더."""

    labeled_dataset: Dataset[dict[str, Any]]
    unlabeled_dataset: Dataset[dict[str, str]]
    tokenizer: Any
    labeled_batch_size: int
    unlabeled_batch_size_multiplier: int
    max_length: int
    shuffle: bool
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.labeled_batch_size <= 0:
            raise ValueError("labeled_batch_size must be positive.")
        if self.unlabeled_batch_size_multiplier <= 0:
            raise ValueError("unlabeled_batch_size_multiplier must be positive.")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        if len(self.labeled_dataset) == 0:
            raise ValueError("labeled_dataset must not be empty.")
        if len(self.unlabeled_dataset) == 0:
            raise ValueError("unlabeled_dataset must not be empty.")

    @property
    def unlabeled_batch_size(self) -> int:
        return self.labeled_batch_size * self.unlabeled_batch_size_multiplier

    def __len__(self) -> int:
        return max(
            math.ceil(len(self.labeled_dataset) / self.labeled_batch_size),
            math.ceil(len(self.unlabeled_dataset) / self.unlabeled_batch_size),
        )

    def __iter__(self) -> Iterator[SemiSupervisedTextBatch]:
        labeled_loader = self._build_labeled_loader()
        unlabeled_loader = self._build_unlabeled_loader()
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        for _ in range(len(self)):
            labeled_batch = _next_or_restart(
                loader=labeled_loader,
                iterator=labeled_iter,
            )
            labeled_iter = labeled_batch.iterator
            unlabeled_batch = _next_or_restart(
                loader=unlabeled_loader,
                iterator=unlabeled_iter,
            )
            unlabeled_iter = unlabeled_batch.iterator
            yield SemiSupervisedTextBatch(
                labeled_query_ids=tuple(labeled_batch.batch["query_ids"]),
                unlabeled_query_ids=tuple(unlabeled_batch.batch["query_ids"]),
                labeled_input_ids=labeled_batch.batch["input_ids"],
                labeled_attention_mask=labeled_batch.batch["attention_mask"],
                labeled_labels=labeled_batch.batch["labels"],
                weak_unlabeled_input_ids=unlabeled_batch.batch["weak_input_ids"],
                weak_unlabeled_attention_mask=unlabeled_batch.batch[
                    "weak_attention_mask"
                ],
                strong_unlabeled_input_ids=unlabeled_batch.batch[
                    "strong_input_ids"
                ],
                strong_unlabeled_attention_mask=unlabeled_batch.batch[
                    "strong_attention_mask"
                ],
            )

    def _build_labeled_loader(self) -> DataLoader[dict[str, Any]]:
        return DataLoader(
            self.labeled_dataset,
            batch_size=self.labeled_batch_size,
            shuffle=self.shuffle,
            collate_fn=self._collate_labeled,
            generator=_build_torch_generator(self.seed),
        )

    def _build_unlabeled_loader(self) -> DataLoader[dict[str, Any]]:
        return DataLoader(
            self.unlabeled_dataset,
            batch_size=self.unlabeled_batch_size,
            shuffle=self.shuffle,
            collate_fn=self._collate_unlabeled,
            generator=_build_torch_generator(self.seed),
        )

    def _collate_labeled(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        texts = [str(item["text"]) for item in batch]
        labels = [int(item["label"]) for item in batch]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        encoded["query_ids"] = [str(item["query_id"]) for item in batch]
        return encoded

    def _collate_unlabeled(
        self,
        batch: list[dict[str, str]],
    ) -> dict[str, Any]:
        weak_encoded = self.tokenizer(
            [item["weak_text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        strong_encoded = self.tokenizer(
            [item["strong_text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "query_ids": [item["query_id"] for item in batch],
            "weak_input_ids": weak_encoded["input_ids"],
            "weak_attention_mask": weak_encoded["attention_mask"],
            "strong_input_ids": strong_encoded["input_ids"],
            "strong_attention_mask": strong_encoded["attention_mask"],
        }


def build_semi_supervised_text_batch_loader(
    *,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[Mapping[str, object]],
    label_to_index: Mapping[str, int],
    tokenizer: Any,
    labeled_batch_size: int,
    unlabeled_batch_size_multiplier: int,
    max_length: int,
    task_prefix: str,
    shuffle: bool,
    seed: int | None = None,
) -> SemiSupervisedTextBatchLoader:
    """labeled + weak/strong unlabeled를 한 번에 순회하는 로더를 만든다."""

    return SemiSupervisedTextBatchLoader(
        labeled_dataset=_LabeledTextDataset(
            rows=labeled_rows,
            label_to_index=label_to_index,
            task_prefix=task_prefix,
        ),
        unlabeled_dataset=_UnlabeledMultiviewTextDataset(
            rows=unlabeled_rows,
            task_prefix=task_prefix,
        ),
        tokenizer=tokenizer,
        labeled_batch_size=labeled_batch_size,
        unlabeled_batch_size_multiplier=unlabeled_batch_size_multiplier,
        max_length=max_length,
        shuffle=shuffle,
        seed=seed,
    )


@dataclass(slots=True)
class _LoaderStep:
    iterator: Iterator[dict[str, Any]]
    batch: dict[str, Any]


def _next_or_restart(
    *,
    loader: Iterable[dict[str, Any]],
    iterator: Iterator[dict[str, Any]],
) -> _LoaderStep:
    try:
        return _LoaderStep(iterator=iterator, batch=next(iterator))
    except StopIteration:
        restarted_iterator = iter(loader)
        return _LoaderStep(iterator=restarted_iterator, batch=next(restarted_iterator))


def _require_row_str(
    row: Mapping[str, object],
    field_name: str,
) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _build_torch_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


__all__ = [
    "SemiSupervisedTextBatch",
    "SemiSupervisedTextBatchLoader",
    "build_semi_supervised_text_batch_loader",
]

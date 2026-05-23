"""Query adaptation supervised / Query SSL baseline용 데이터 준비 유틸리티."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from methods.adaptation.query_classifier_adaptation.tokenization import (
    TextTokenizationCache,
    encode_texts,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


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
    """strict USB aug candidate 또는 legacy weak/strong row를 배치용으로 노출한다."""

    def __init__(
        self,
        *,
        rows: list[LabeledQueryRow],
        task_prefix: str,
        strong_view_policy: str = "first_aug",
    ) -> None:
        self._rows = rows
        self._task_prefix = task_prefix
        self._strong_view_policy = _normalize_strong_view_policy(strong_view_policy)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        aug_0 = row.get("aug_0")
        aug_1 = row.get("aug_1")
        if aug_0 is not None and aug_1 is not None:
            weak_text = str(row["text"])
            strong_text = _select_usb_strong_view(
                aug_0=str(aug_0),
                aug_1=str(aug_1),
                row_index=index,
                query_id=str(row["query_id"]),
                policy=self._strong_view_policy,
            )
        else:
            weak_text = row.get("weak_text")
            strong_text = row.get("strong_text")
            if weak_text is None or strong_text is None:
                raise ValueError(
                    "Multiview Query SSL unlabeled rows require either strict USB "
                    "text/aug_0/aug_1 fields or legacy weak_text/strong_text."
                )
            weak_text = str(weak_text)
            strong_text = str(strong_text)
        if self._task_prefix:
            weak_text = f"{self._task_prefix}{weak_text}"
            strong_text = f"{self._task_prefix}{strong_text}"
        return {
            "query_id": str(row["query_id"]),
            "row_index": int(index),
            "weak_text": weak_text,
            "strong_text": strong_text,
        }


class TextWeakDataset(Dataset[dict[str, Any]]):
    """USB PseudoLabel처럼 weak/original unlabeled view만 배치로 노출한다."""

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
        weak_text = str(row.get("weak_text") or row["text"])
        if self._task_prefix:
            weak_text = f"{self._task_prefix}{weak_text}"
        return {
            "query_id": str(row["query_id"]),
            "row_index": int(index),
            "weak_text": weak_text,
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
    tokenization_cache: TextTokenizationCache | None = None,
    tokenization_cache_namespace: str | None = None,
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
        encoded = encode_texts(
            texts,
            tokenizer=tokenizer,
            max_length=max_length,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace,
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        pin_memory=_should_pin_memory(),
    )


def build_weak_dataloader(
    *,
    rows: list[LabeledQueryRow],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    task_prefix: str,
    shuffle: bool,
    tokenization_cache: TextTokenizationCache | None = None,
    tokenization_cache_namespace: str | None = None,
) -> DataLoader[dict[str, Any]]:
    """weak/original unlabeled row를 Query SSL 입력 DataLoader로 변환한다."""

    dataset = TextWeakDataset(
        rows=rows,
        task_prefix=task_prefix,
    )

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        weak_texts = [str(item["weak_text"]) for item in batch]
        weak_encoded = encode_texts(
            weak_texts,
            tokenizer=tokenizer,
            max_length=max_length,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace,
        )
        return {
            "query_ids": [str(item["query_id"]) for item in batch],
            "row_indices": torch.tensor(
                [int(item["row_index"]) for item in batch], dtype=torch.long
            ),
            "weak_input_ids": weak_encoded["input_ids"],
            "weak_attention_mask": weak_encoded["attention_mask"],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        pin_memory=_should_pin_memory(),
    )


def build_multiview_dataloader(
    *,
    rows: list[LabeledQueryRow],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    task_prefix: str,
    shuffle: bool,
    strong_view_policy: str = "first_aug",
    tokenization_cache: TextTokenizationCache | None = None,
    tokenization_cache_namespace: str | None = None,
) -> DataLoader[dict[str, Any]]:
    """weak/strong unlabeled row를 Query SSL 입력 DataLoader로 변환한다."""

    dataset = TextMultiviewDataset(
        rows=rows,
        task_prefix=task_prefix,
        strong_view_policy=strong_view_policy,
    )

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        weak_texts = [str(item["weak_text"]) for item in batch]
        strong_texts = [str(item["strong_text"]) for item in batch]
        weak_encoded = encode_texts(
            weak_texts,
            tokenizer=tokenizer,
            max_length=max_length,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace,
        )
        strong_encoded = encode_texts(
            strong_texts,
            tokenizer=tokenizer,
            max_length=max_length,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace,
        )
        return {
            "query_ids": [str(item["query_id"]) for item in batch],
            "row_indices": torch.tensor(
                [int(item["row_index"]) for item in batch], dtype=torch.long
            ),
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
        pin_memory=_should_pin_memory(),
    )


def _should_pin_memory() -> bool:
    return bool(torch.cuda.is_available())


def _normalize_strong_view_policy(value: str) -> str:
    policy = str(value).strip()
    if not policy:
        raise ValueError("strong_view_policy must not be empty.")
    supported = {
        "first_aug",
        "second_aug",
        "row_parity_aug",
        "query_id_hash_aug",
    }
    if policy not in supported:
        raise ValueError(
            "Unsupported strong_view_policy. "
            f"Expected one of {sorted(supported)}, got {policy!r}."
        )
    return policy


def _select_usb_strong_view(
    *,
    aug_0: str,
    aug_1: str,
    row_index: int,
    query_id: str,
    policy: str,
) -> str:
    if policy == "first_aug":
        return aug_0
    if policy == "second_aug":
        return aug_1
    if policy == "row_parity_aug":
        return aug_0 if row_index % 2 == 0 else aug_1
    if policy == "query_id_hash_aug":
        digest = hashlib.sha256(query_id.encode("utf-8")).digest()
        return aug_0 if digest[0] % 2 == 0 else aug_1
    raise ValueError(f"Unsupported strong_view_policy: {policy!r}.")

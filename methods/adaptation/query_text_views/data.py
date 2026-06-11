"""Query adaptation supervised / Query SSL baseline용 데이터 준비 유틸리티."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from methods.adaptation.query_text_views.tokenization import (
    TextTokenizationCache,
    encode_texts,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

STRONG_VIEW_FIRST_AUG = "first_aug"
STRONG_VIEW_SECOND_AUG = "second_aug"
STRONG_VIEW_ROW_PARITY_AUG = "row_parity_aug"
STRONG_VIEW_QUERY_ID_HASH_AUG = "query_id_hash_aug"
DEFAULT_STRONG_VIEW_POLICY = STRONG_VIEW_FIRST_AUG
STRONG_VIEW_POLICIES = frozenset(
    {
        STRONG_VIEW_FIRST_AUG,
        STRONG_VIEW_SECOND_AUG,
        STRONG_VIEW_ROW_PARITY_AUG,
        STRONG_VIEW_QUERY_ID_HASH_AUG,
    }
)


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

    def label_histogram(self, *, num_classes: int) -> torch.Tensor:
        """labeled dataset 전체의 class 분포 계산용 label count를 반환한다."""

        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        counts = torch.zeros((num_classes,), dtype=torch.float32)
        for row in self._rows:
            label = self._label_to_index[str(row["mapped_label_4"])]
            if label < 0 or label >= num_classes:
                raise ValueError("label index is outside num_classes.")
            counts[label] += 1.0
        return counts

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        text = str(row["text"])
        if self._task_prefix:
            text = f"{self._task_prefix}{text}"
        return {
            "row_index": int(index),
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
        strong_view_policy: str = DEFAULT_STRONG_VIEW_POLICY,
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


class TextWeakStrongPairDataset(Dataset[dict[str, Any]]):
    """weak view와 USB strong 후보 2개를 모두 배치용으로 노출한다."""

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
        aug_0 = row.get("aug_0")
        aug_1 = row.get("aug_1")
        if aug_0 is None or aug_1 is None:
            raise ValueError(
                "Weak/strong-pair Query SSL rows require strict USB "
                "text/aug_0/aug_1 fields."
            )
        weak_text = str(row["text"])
        strong_0_text = str(aug_0)
        strong_1_text = str(aug_1)
        if self._task_prefix:
            weak_text = f"{self._task_prefix}{weak_text}"
            strong_0_text = f"{self._task_prefix}{strong_0_text}"
            strong_1_text = f"{self._task_prefix}{strong_1_text}"
        return {
            "query_id": str(row["query_id"]),
            "row_index": int(index),
            "weak_text": weak_text,
            "strong_0_text": strong_0_text,
            "strong_1_text": strong_1_text,
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
    drop_last: bool = False,
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
        encoded["row_indices"] = torch.tensor(
            [int(item["row_index"]) for item in batch],
            dtype=torch.long,
        )
        return encoded

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        pin_memory=_should_pin_memory(),
        drop_last=drop_last,
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
    drop_last: bool = False,
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
        drop_last=drop_last,
    )


def build_multiview_dataloader(
    *,
    rows: list[LabeledQueryRow],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    task_prefix: str,
    shuffle: bool,
    strong_view_policy: str = DEFAULT_STRONG_VIEW_POLICY,
    tokenization_cache: TextTokenizationCache | None = None,
    tokenization_cache_namespace: str | None = None,
    drop_last: bool = False,
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
        drop_last=drop_last,
    )


def build_weak_strong_pair_dataloader(
    *,
    rows: list[LabeledQueryRow],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    task_prefix: str,
    shuffle: bool,
    tokenization_cache: TextTokenizationCache | None = None,
    tokenization_cache_namespace: str | None = None,
    drop_last: bool = False,
) -> DataLoader[dict[str, Any]]:
    """weak/strong_0/strong_1 row를 Query SSL 입력 DataLoader로 변환한다."""

    dataset = TextWeakStrongPairDataset(
        rows=rows,
        task_prefix=task_prefix,
    )

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        weak_texts = [str(item["weak_text"]) for item in batch]
        strong_0_texts = [str(item["strong_0_text"]) for item in batch]
        strong_1_texts = [str(item["strong_1_text"]) for item in batch]
        weak_encoded = encode_texts(
            weak_texts,
            tokenizer=tokenizer,
            max_length=max_length,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace,
        )
        strong_0_encoded = encode_texts(
            strong_0_texts,
            tokenizer=tokenizer,
            max_length=max_length,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace,
        )
        strong_1_encoded = encode_texts(
            strong_1_texts,
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
            "strong_0_input_ids": strong_0_encoded["input_ids"],
            "strong_0_attention_mask": strong_0_encoded["attention_mask"],
            "strong_1_input_ids": strong_1_encoded["input_ids"],
            "strong_1_attention_mask": strong_1_encoded["attention_mask"],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        pin_memory=_should_pin_memory(),
        drop_last=drop_last,
    )


def _should_pin_memory() -> bool:
    return bool(torch.cuda.is_available())


def _normalize_strong_view_policy(value: str) -> str:
    policy = str(value).strip()
    if not policy:
        raise ValueError("strong_view_policy must not be empty.")
    if policy not in STRONG_VIEW_POLICIES:
        raise ValueError(
            "Unsupported strong_view_policy. "
            f"Expected one of {sorted(STRONG_VIEW_POLICIES)}, got {policy!r}."
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
    if policy == STRONG_VIEW_FIRST_AUG:
        return aug_0
    if policy == STRONG_VIEW_SECOND_AUG:
        return aug_1
    if policy == STRONG_VIEW_ROW_PARITY_AUG:
        return aug_0 if row_index % 2 == 0 else aug_1
    if policy == STRONG_VIEW_QUERY_ID_HASH_AUG:
        digest = hashlib.sha256(query_id.encode("utf-8")).digest()
        return aug_0 if digest[0] % 2 == 0 else aug_1
    raise ValueError(f"Unsupported strong_view_policy: {policy!r}.")

"""LabeledQueryRow를 fixed embedding tensor로 변환하는 helper."""

from __future__ import annotations

import torch

from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter


def batched_rows(
    rows: list[LabeledQueryRow],
    chunk_size: int,
) -> list[list[LabeledQueryRow]]:
    return [
        rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)
    ]


def embed_rows(
    *,
    rows: list[LabeledQueryRow],
    adapter: EmbeddingAdapter,
    chunk_size: int,
) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for chunk in batched_rows(rows, chunk_size):
        texts = [row["text"] for row in chunk]
        embeddings = adapter.embed_texts(texts)
        tensors.append(torch.tensor(embeddings, dtype=torch.float32))
    return torch.cat(tensors, dim=0)

"""Query SSL unlabeled view DataLoader 공통 builder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from methods.adaptation.query_text_views.data import (
    DEFAULT_STRONG_VIEW_POLICY,
    build_multiview_dataloader,
    build_weak_dataloader,
    build_weak_strong_pair_dataloader,
)
from methods.adaptation.query_text_views.tokenization import (
    TextTokenizationCache,
)
from methods.adaptation.query_text_views.view_rows import (
    USB_MULTIVIEW_BUILDER_NAME,
    USB_WEAK_BUILDER_NAME,
    USB_WEAK_STRONG_PAIR_BUILDER_NAME,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


def build_query_ssl_unlabeled_dataloader(
    *,
    rows: Sequence[LabeledQueryRow],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    task_prefix: str,
    shuffle: bool,
    view_builder_name: str,
    strong_view_policy: str = DEFAULT_STRONG_VIEW_POLICY,
    tokenization_cache: TextTokenizationCache | None = None,
    tokenization_cache_namespace: str | None = None,
) -> Any:
    """Query SSL view surface 이름으로 unlabeled DataLoader를 만든다."""

    effective_rows = list(rows)
    if view_builder_name == USB_MULTIVIEW_BUILDER_NAME:
        return build_multiview_dataloader(
            rows=effective_rows,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            task_prefix=task_prefix,
            shuffle=shuffle,
            strong_view_policy=strong_view_policy,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace,
        )
    if view_builder_name == USB_WEAK_STRONG_PAIR_BUILDER_NAME:
        return build_weak_strong_pair_dataloader(
            rows=effective_rows,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            task_prefix=task_prefix,
            shuffle=shuffle,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace,
        )
    if view_builder_name == USB_WEAK_BUILDER_NAME:
        return build_weak_dataloader(
            rows=effective_rows,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            task_prefix=task_prefix,
            shuffle=shuffle,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace,
        )
    raise ValueError(f"Unsupported Query SSL view builder: {view_builder_name}.")

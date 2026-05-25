"""중앙 Query SSL unlabeled view 준비 resolver."""

from __future__ import annotations

from pathlib import Path

from methods.ssl.base import QuerySslAlgorithmDescriptor
from scripts.experiments.query_lora_ssl.query_ssl.augmentation import (
    PreparedQuerySslUnlabeledRows,
    prepare_usb_multiview_unlabeled_rows,
    prepare_usb_weak_unlabeled_rows,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


def prepare_query_ssl_unlabeled_rows(
    *,
    cfg: object,
    descriptor: QuerySslAlgorithmDescriptor,
    rows: list[LabeledQueryRow],
    source_jsonl: str | Path | None,
) -> PreparedQuerySslUnlabeledRows:
    """algorithm descriptor의 view surface에 맞춰 중앙 unlabeled rows를 준비한다."""

    if descriptor.required_views.view_builder_name == "usb_multiview":
        return prepare_usb_multiview_unlabeled_rows(
            cfg,
            rows=rows,
            source_jsonl=source_jsonl,
            algorithm_name=descriptor.display_name,
        )
    if descriptor.required_views.view_builder_name == "usb_weak":
        return prepare_usb_weak_unlabeled_rows(
            rows=rows,
            algorithm_name=descriptor.display_name,
        )
    raise ValueError(
        "Unsupported Query SSL view builder: "
        f"{descriptor.required_views.view_builder_name}."
    )

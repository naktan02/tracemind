"""dataset source별 download adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.workflows.datasets.lib.download import (
    download_huggingface_dataset_to_csv,
    download_kaggle_dataset_file_to_csv,
)


def download_huggingface_source(
    *,
    source_name: str,
    source_cfg: Any,
    raw_dir: Path,
    cache_dir: Path,
    output_path: Path,
) -> Path:
    """HuggingFace source config 하나를 raw CSV로 materialize한다."""

    return download_huggingface_dataset_to_csv(
        dataset_id=_require_string(
            source_cfg.get("dataset_id"),
            field_name=f"sources.{source_name}.dataset_id",
        ),
        split=_require_string(
            source_cfg.get("split"),
            field_name=f"sources.{source_name}.split",
        ),
        output_dir=raw_dir,
        cache_dir=cache_dir,
        data_file=source_cfg.get("data_file"),
        output_path=output_path,
        revision=source_cfg.get("revision"),
    )


def download_kaggle_source(
    *,
    source_name: str,
    source_cfg: Any,
    raw_dir: Path,
    cache_dir: Path,
    output_path: Path,
) -> Path:
    """Kaggle source config 하나를 raw CSV로 materialize한다."""

    del raw_dir, cache_dir
    return download_kaggle_dataset_file_to_csv(
        dataset_ref=_require_string(
            source_cfg.get("dataset_ref") or source_cfg.get("dataset_id"),
            field_name=f"sources.{source_name}.dataset_ref",
        ),
        data_file=_require_string(
            source_cfg.get("data_file"),
            field_name=f"sources.{source_name}.data_file",
        ),
        output_path=output_path,
        dataset_version_number=source_cfg.get("dataset_version_number"),
        download_url=source_cfg.get("download_url"),
    )


def _require_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string field: {field_name}")
    return value

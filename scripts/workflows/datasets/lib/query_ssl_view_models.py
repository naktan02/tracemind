"""Query SSL view materialization models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

QUERY_SSL_VIEWS_SCHEMA_VERSION = "query_ssl_views.v1"
QUERY_SSL_VIEWS_PROGRESS_SCHEMA_VERSION = "query_ssl_views_progress.v1"


@dataclass(frozen=True, slots=True)
class QuerySslViewArtifacts:
    """Query SSL view materializer가 쓰는 산출물 경로."""

    labeled_train_with_views_jsonl: Path
    unlabeled_pool_with_views_jsonl: Path
    manifest_json: Path
    summary_json: Path
    progress_json: Path


@dataclass(frozen=True, slots=True)
class ViewPartitionArtifacts:
    partition_name: str
    source_jsonl: Path
    final_jsonl: Path
    tmp_jsonl: Path


@dataclass(frozen=True, slots=True)
class ViewPartitionResult:
    partition_name: str
    rows: list[LabeledQueryRow]
    resumed_from_count: int


@dataclass(frozen=True, slots=True)
class NllbBacktranslationRuntimeConfig:
    """NLLB backtranslation runtime 설정."""

    source_lang: str
    pivot_languages: tuple[str, str]
    model_id: str
    revision: str
    device: str
    batch_size: int
    max_new_tokens: int
    torch_dtype: str
    cache_dir: str | None
    local_files_only: bool

    def to_manifest(self) -> dict[str, object]:
        return {
            "augmenter_type": "nllb_backtranslation",
            "source_lang": self.source_lang,
            "pivot_languages": list(self.pivot_languages),
            "model_id": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_new_tokens": self.max_new_tokens,
            "torch_dtype": self.torch_dtype,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }

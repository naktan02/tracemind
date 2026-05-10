"""중앙 Query SSL용 weak/strong text view materialization."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    count_labeled_query_rows_by_label,
    dump_labeled_query_rows,
    load_labeled_query_rows,
)

QUERY_SSL_VIEWS_SCHEMA_VERSION = "query_ssl_views.v1"


class QuerySslBacktranslationPair(Protocol):
    """원문 하나에서 생성한 두 개의 strong view."""

    aug_0: str
    aug_1: str
    aug_0_pivot_lang: str
    aug_1_pivot_lang: str


QuerySslCandidatePairBuilder = Callable[
    [Sequence[str]],
    Sequence[QuerySslBacktranslationPair],
]


@dataclass(frozen=True, slots=True)
class QuerySslViewArtifacts:
    """Query SSL view materializer가 쓰는 산출물 경로."""

    labeled_train_with_views_jsonl: Path
    unlabeled_pool_with_views_jsonl: Path
    manifest_json: Path
    summary_json: Path


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


def materialize_query_ssl_backtranslation_views(
    *,
    split_dir: Path,
    split_name: str,
    augmenter_name: str,
    output_root: Path,
    augmenter_manifest: dict[str, object],
    candidate_pair_builder: QuerySslCandidatePairBuilder,
) -> QuerySslViewArtifacts:
    """labeled/unlabeled rows에 `aug_0`, `aug_1` strong view를 저장한다."""

    normalized_split_name = split_name.strip()
    normalized_augmenter_name = augmenter_name.strip()
    if not normalized_split_name:
        raise ValueError("split_name must not be empty.")
    if not normalized_augmenter_name:
        raise ValueError("augmenter_name must not be empty.")

    labeled_rows = load_labeled_query_rows(split_dir / "labeled_train.jsonl")
    unlabeled_rows = load_labeled_query_rows(split_dir / "unlabeled_pool.jsonl")
    labeled_with_views = _attach_backtranslation_views(
        rows=labeled_rows,
        candidate_pair_builder=candidate_pair_builder,
    )
    unlabeled_with_views = _attach_backtranslation_views(
        rows=unlabeled_rows,
        candidate_pair_builder=candidate_pair_builder,
    )

    output_dir = output_root / normalized_split_name / normalized_augmenter_name
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = QuerySslViewArtifacts(
        labeled_train_with_views_jsonl=output_dir / "labeled_train.with_views.jsonl",
        unlabeled_pool_with_views_jsonl=output_dir / "unlabeled_pool.with_views.jsonl",
        manifest_json=output_dir / "manifest.json",
        summary_json=output_dir / "summary.json",
    )
    dump_labeled_query_rows(
        artifacts.labeled_train_with_views_jsonl, labeled_with_views
    )
    dump_labeled_query_rows(
        artifacts.unlabeled_pool_with_views_jsonl,
        unlabeled_with_views,
    )

    manifest = _build_query_ssl_views_manifest(
        split_dir=split_dir,
        split_name=normalized_split_name,
        augmenter_name=normalized_augmenter_name,
        augmenter_manifest=augmenter_manifest,
        labeled_rows=labeled_with_views,
        unlabeled_rows=unlabeled_with_views,
        artifacts=artifacts,
    )
    artifacts.manifest_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    artifacts.summary_json.write_text(
        json.dumps(
            _build_query_ssl_views_summary(manifest), indent=2, ensure_ascii=True
        )
        + "\n",
        encoding="utf-8",
    )
    return artifacts


def build_nllb_backtranslation_candidate_pair_builder(
    config: NllbBacktranslationRuntimeConfig,
) -> QuerySslCandidatePairBuilder:
    """NLLB runtime config로 candidate pair builder를 만든다."""

    def _build(texts: Sequence[str]) -> Sequence[QuerySslBacktranslationPair]:
        from agent.src.services.language.backtranslation_service import (
            NllbBacktranslationService,
        )

        service = NllbBacktranslationService(
            source_lang=config.source_lang,
            pivot_languages=config.pivot_languages,
            model_id=config.model_id,
            revision=config.revision,
            device=config.device,
            batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens,
            torch_dtype=config.torch_dtype,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        )
        return service.build_candidate_pairs(texts=list(texts))

    return _build


def _attach_backtranslation_views(
    *,
    rows: Sequence[LabeledQueryRow],
    candidate_pair_builder: QuerySslCandidatePairBuilder,
) -> list[LabeledQueryRow]:
    candidate_pairs = list(candidate_pair_builder([str(row["text"]) for row in rows]))
    if len(candidate_pairs) != len(rows):
        raise ValueError(
            "Backtranslation candidate count does not match source row count."
        )

    rows_with_views: list[LabeledQueryRow] = []
    for row, candidate_pair in zip(rows, candidate_pairs):
        row_with_views: LabeledQueryRow = dict(row)  # type: ignore[assignment]
        row_with_views["aug_0"] = candidate_pair.aug_0
        row_with_views["aug_1"] = candidate_pair.aug_1
        row_with_views["aug_0_pivot_lang"] = candidate_pair.aug_0_pivot_lang
        row_with_views["aug_1_pivot_lang"] = candidate_pair.aug_1_pivot_lang
        rows_with_views.append(row_with_views)

    _validate_view_rows(rows_with_views)
    return rows_with_views


def _validate_view_rows(rows: Sequence[LabeledQueryRow]) -> None:
    missing_query_ids = [
        str(row["query_id"])
        for row in rows
        if (
            not str(row.get("aug_0", "")).strip()
            or not str(row.get("aug_1", "")).strip()
        )
    ]
    if missing_query_ids:
        raise ValueError(
            "Query SSL view materialization requires non-empty aug_0 and aug_1. "
            f"Missing examples: {missing_query_ids[:5]}."
        )


def _build_query_ssl_views_manifest(
    *,
    split_dir: Path,
    split_name: str,
    augmenter_name: str,
    augmenter_manifest: dict[str, object],
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    artifacts: QuerySslViewArtifacts,
) -> dict[str, object]:
    return {
        "schema_version": QUERY_SSL_VIEWS_SCHEMA_VERSION,
        "split_name": split_name,
        "source_split_dir": str(split_dir),
        "source_split_manifest_json": str(split_dir / "manifest.json"),
        "augmenter_name": augmenter_name,
        "augmenter": augmenter_manifest,
        "weak_view_policy": {
            "labeled": "text",
            "unlabeled": "text",
        },
        "strong_view_policy": {
            "labeled": ["aug_0", "aug_1"],
            "unlabeled": ["aug_0", "aug_1"],
        },
        "row_counts": {
            "labeled_train": len(labeled_rows),
            "unlabeled_pool": len(unlabeled_rows),
        },
        "label_counts": {
            "labeled_train": count_labeled_query_rows_by_label(labeled_rows),
            "unlabeled_pool": count_labeled_query_rows_by_label(unlabeled_rows),
        },
        "view_counts": {
            "labeled_train": _count_view_metadata(labeled_rows),
            "unlabeled_pool": _count_view_metadata(unlabeled_rows),
        },
        "artifacts": {
            "labeled_train_with_views_jsonl": str(
                artifacts.labeled_train_with_views_jsonl
            ),
            "unlabeled_pool_with_views_jsonl": str(
                artifacts.unlabeled_pool_with_views_jsonl
            ),
            "manifest_json": str(artifacts.manifest_json),
            "summary_json": str(artifacts.summary_json),
        },
    }


def _build_query_ssl_views_summary(
    manifest: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": manifest["schema_version"],
        "split_name": manifest["split_name"],
        "augmenter_name": manifest["augmenter_name"],
        "row_counts": manifest["row_counts"],
        "label_counts": manifest["label_counts"],
        "view_counts": manifest["view_counts"],
    }


def _count_view_metadata(rows: Sequence[LabeledQueryRow]) -> dict[str, object]:
    aug_0_pivot_counts: Counter[str] = Counter()
    aug_1_pivot_counts: Counter[str] = Counter()
    empty_aug_0_count = 0
    empty_aug_1_count = 0
    for row in rows:
        aug_0_pivot_counts[str(row.get("aug_0_pivot_lang", ""))] += 1
        aug_1_pivot_counts[str(row.get("aug_1_pivot_lang", ""))] += 1
        if not str(row.get("aug_0", "")).strip():
            empty_aug_0_count += 1
        if not str(row.get("aug_1", "")).strip():
            empty_aug_1_count += 1

    return {
        "aug_0_pivot_lang_counts": dict(sorted(aug_0_pivot_counts.items())),
        "aug_1_pivot_lang_counts": dict(sorted(aug_1_pivot_counts.items())),
        "empty_aug_0_count": empty_aug_0_count,
        "empty_aug_1_count": empty_aug_1_count,
    }

"""중앙 Query SSL용 weak/strong text view materialization."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from methods.adaptation.query_classifier_adaptation.view_rows import (
    QuerySslBacktranslationPair,
    attach_usb_multiview_candidate_pair,
    validate_usb_multiview_candidate_rows,
)
from scripts.datasets.lib.query_ssl_view_manifest import (
    build_query_ssl_views_manifest,
    build_query_ssl_views_summary,
)
from scripts.datasets.lib.query_ssl_view_models import (
    QUERY_SSL_VIEWS_SCHEMA_VERSION as _QUERY_SSL_VIEWS_SCHEMA_VERSION,
)
from scripts.datasets.lib.query_ssl_view_models import (
    NllbBacktranslationRuntimeConfig,
    QuerySslViewArtifacts,
    ViewPartitionArtifacts,
    ViewPartitionResult,
)
from scripts.datasets.lib.query_ssl_view_progress import (
    build_initial_progress,
    update_partition_progress,
    utc_now,
    write_progress,
)
from scripts.runtime_adapters.backtranslation_runtime import (
    build_nllb_backtranslation_candidate_pairs_from_params,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    load_labeled_query_rows,
)

QuerySslCandidatePairBuilder = Callable[
    [Sequence[str]],
    Sequence[QuerySslBacktranslationPair],
]
QUERY_SSL_VIEWS_SCHEMA_VERSION = _QUERY_SSL_VIEWS_SCHEMA_VERSION


def materialize_query_ssl_backtranslation_views(
    *,
    split_dir: Path,
    split_name: str,
    augmenter_name: str,
    output_root: Path,
    augmenter_manifest: dict[str, object],
    candidate_pair_builder: QuerySslCandidatePairBuilder,
    chunk_size: int = 256,
    resume: bool = True,
    overwrite: bool = False,
) -> QuerySslViewArtifacts:
    """labeled/unlabeled rows에 `aug_0`, `aug_1` strong view를 저장한다.

    긴 NLLB 작업을 안전하게 재시작할 수 있도록 partition별 `.tmp` JSONL에
    chunk 단위로 append하고, 완료 시 final JSONL로 atomic replace한다.
    """

    normalized_split_name = split_name.strip()
    normalized_augmenter_name = augmenter_name.strip()
    if not normalized_split_name:
        raise ValueError("split_name must not be empty.")
    if not normalized_augmenter_name:
        raise ValueError("augmenter_name must not be empty.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    output_dir = output_root / normalized_split_name / normalized_augmenter_name
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = QuerySslViewArtifacts(
        labeled_train_with_views_jsonl=output_dir / "labeled_train.with_views.jsonl",
        unlabeled_pool_with_views_jsonl=output_dir / "unlabeled_pool.with_views.jsonl",
        manifest_json=output_dir / "manifest.json",
        summary_json=output_dir / "summary.json",
        progress_json=output_dir / "progress.json",
    )
    partitions = (
        ViewPartitionArtifacts(
            partition_name="labeled_train",
            source_jsonl=split_dir / "labeled_train.jsonl",
            final_jsonl=artifacts.labeled_train_with_views_jsonl,
            tmp_jsonl=artifacts.labeled_train_with_views_jsonl.with_suffix(
                ".jsonl.tmp"
            ),
        ),
        ViewPartitionArtifacts(
            partition_name="unlabeled_pool",
            source_jsonl=split_dir / "unlabeled_pool.jsonl",
            final_jsonl=artifacts.unlabeled_pool_with_views_jsonl,
            tmp_jsonl=artifacts.unlabeled_pool_with_views_jsonl.with_suffix(
                ".jsonl.tmp"
            ),
        ),
    )

    if overwrite:
        _remove_existing_view_artifacts(artifacts=artifacts, partitions=partitions)
    elif _final_artifacts_exist(artifacts):
        return artifacts

    progress = build_initial_progress(
        split_dir=split_dir,
        split_name=normalized_split_name,
        augmenter_name=normalized_augmenter_name,
        augmenter_manifest=augmenter_manifest,
        chunk_size=chunk_size,
        artifacts=artifacts,
        partitions=partitions,
    )
    write_progress(artifacts.progress_json, progress)

    partition_results: list[ViewPartitionResult] = []
    for partition in partitions:
        partition_results.append(
            _materialize_view_partition(
                partition=partition,
                progress_path=artifacts.progress_json,
                progress=progress,
                candidate_pair_builder=candidate_pair_builder,
                chunk_size=chunk_size,
                resume=resume,
            )
        )

    labeled_rows = partition_results[0].rows
    unlabeled_rows = partition_results[1].rows
    manifest = build_query_ssl_views_manifest(
        split_dir=split_dir,
        split_name=normalized_split_name,
        augmenter_name=normalized_augmenter_name,
        augmenter_manifest=augmenter_manifest,
        chunk_size=chunk_size,
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        partition_results=partition_results,
        artifacts=artifacts,
    )
    artifacts.manifest_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    artifacts.summary_json.write_text(
        json.dumps(build_query_ssl_views_summary(manifest), indent=2, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    progress["status"] = "completed"
    progress["completed_at"] = utc_now()
    progress["manifest_json"] = str(artifacts.manifest_json)
    progress["summary_json"] = str(artifacts.summary_json)
    write_progress(artifacts.progress_json, progress)
    return artifacts


def build_nllb_backtranslation_candidate_pair_builder(
    config: NllbBacktranslationRuntimeConfig,
) -> QuerySslCandidatePairBuilder:
    """NLLB runtime config로 candidate pair builder를 만든다."""

    def _build(texts: Sequence[str]) -> Sequence[QuerySslBacktranslationPair]:
        return build_nllb_backtranslation_candidate_pairs_from_params(
            texts=texts,
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

    return _build


def _materialize_view_partition(
    *,
    partition: ViewPartitionArtifacts,
    progress_path: Path,
    progress: dict[str, object],
    candidate_pair_builder: QuerySslCandidatePairBuilder,
    chunk_size: int,
    resume: bool,
) -> ViewPartitionResult:
    source_rows = load_labeled_query_rows(partition.source_jsonl)
    if partition.final_jsonl.exists():
        final_rows = load_labeled_query_rows(partition.final_jsonl)
        if len(final_rows) != len(source_rows):
            raise ValueError(
                f"{partition.final_jsonl} exists but has {len(final_rows)} rows; "
                f"expected {len(source_rows)}. Use --overwrite to regenerate."
            )
        update_partition_progress(
            progress=progress,
            partition=partition,
            total_count=len(source_rows),
            processed_count=len(final_rows),
            status="completed",
        )
        write_progress(progress_path, progress)
        return ViewPartitionResult(
            partition_name=partition.partition_name,
            rows=final_rows,
            resumed_from_count=len(final_rows),
        )

    processed_ids = (
        _load_existing_tmp_query_ids(partition.tmp_jsonl) if resume else set()
    )
    if processed_ids and not _source_prefix_matches_tmp(
        source_rows=source_rows,
        processed_ids=processed_ids,
    ):
        raise ValueError(
            f"{partition.tmp_jsonl} does not match the source row prefix. "
            "Use --overwrite to regenerate."
        )
    if not resume and partition.tmp_jsonl.exists():
        partition.tmp_jsonl.unlink()
        processed_ids = set()

    resumed_from_count = len(processed_ids)
    update_partition_progress(
        progress=progress,
        partition=partition,
        total_count=len(source_rows),
        processed_count=resumed_from_count,
        status="running",
    )
    write_progress(progress_path, progress)

    if not source_rows:
        partition.tmp_jsonl.parent.mkdir(parents=True, exist_ok=True)
        partition.tmp_jsonl.write_text("", encoding="utf-8")

    pending_rows = [
        row for row in source_rows if str(row["query_id"]) not in processed_ids
    ]
    for chunk in _iter_chunks(pending_rows, chunk_size):
        rows_with_views = _attach_backtranslation_views(
            rows=chunk,
            candidate_pair_builder=candidate_pair_builder,
        )
        _append_labeled_query_rows(partition.tmp_jsonl, rows_with_views)
        processed_ids.update(str(row["query_id"]) for row in rows_with_views)
        update_partition_progress(
            progress=progress,
            partition=partition,
            total_count=len(source_rows),
            processed_count=len(processed_ids),
            status="running",
        )
        write_progress(progress_path, progress)
        print(
            "query_ssl_views=chunk_complete "
            f"partition={partition.partition_name} "
            f"processed={len(processed_ids)}/{len(source_rows)} "
            f"tmp_jsonl={partition.tmp_jsonl}",
            flush=True,
        )

    tmp_rows = load_labeled_query_rows(partition.tmp_jsonl)
    if len(tmp_rows) != len(source_rows):
        raise ValueError(
            f"{partition.tmp_jsonl} has {len(tmp_rows)} rows after materialization; "
            f"expected {len(source_rows)}."
        )
    partition.tmp_jsonl.replace(partition.final_jsonl)
    update_partition_progress(
        progress=progress,
        partition=partition,
        total_count=len(source_rows),
        processed_count=len(source_rows),
        status="completed",
    )
    write_progress(progress_path, progress)
    print(
        "query_ssl_views=partition_complete "
        f"partition={partition.partition_name} "
        f"rows={len(source_rows)} "
        f"final_jsonl={partition.final_jsonl}",
        flush=True,
    )
    return ViewPartitionResult(
        partition_name=partition.partition_name,
        rows=tmp_rows,
        resumed_from_count=resumed_from_count,
    )


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
        rows_with_views.append(attach_usb_multiview_candidate_pair(row, candidate_pair))

    validate_usb_multiview_candidate_rows(
        rows_with_views,
        context="Query SSL view materialization",
    )
    return rows_with_views


def _append_labeled_query_rows(
    path: Path,
    rows: Sequence[LabeledQueryRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_existing_tmp_query_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    query_ids: set[str] = set()
    for row in load_labeled_query_rows(path):
        validate_usb_multiview_candidate_rows(
            [row],
            context="Query SSL view materialization",
        )
        query_id = str(row["query_id"])
        if query_id in query_ids:
            raise ValueError(f"{path} contains duplicate query_id={query_id!r}.")
        query_ids.add(query_id)
    return query_ids


def _source_prefix_matches_tmp(
    *,
    source_rows: Sequence[LabeledQueryRow],
    processed_ids: set[str],
) -> bool:
    expected_prefix_ids = {
        str(row["query_id"]) for row in source_rows[: len(processed_ids)]
    }
    return expected_prefix_ids == processed_ids


def _iter_chunks(
    rows: Sequence[LabeledQueryRow],
    chunk_size: int,
) -> Sequence[Sequence[LabeledQueryRow]]:
    return [
        rows[start_index : start_index + chunk_size]
        for start_index in range(0, len(rows), chunk_size)
    ]


def _remove_existing_view_artifacts(
    *,
    artifacts: QuerySslViewArtifacts,
    partitions: Sequence[ViewPartitionArtifacts],
) -> None:
    for path in (
        artifacts.manifest_json,
        artifacts.summary_json,
        artifacts.progress_json,
    ):
        if path.exists():
            path.unlink()
    for partition in partitions:
        for path in (partition.final_jsonl, partition.tmp_jsonl):
            if path.exists():
                path.unlink()


def _final_artifacts_exist(artifacts: QuerySslViewArtifacts) -> bool:
    return (
        artifacts.labeled_train_with_views_jsonl.exists()
        and artifacts.unlabeled_pool_with_views_jsonl.exists()
        and artifacts.manifest_json.exists()
        and artifacts.summary_json.exists()
    )

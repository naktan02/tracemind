"""Query SSL unlabeled row view preparation core."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from methods.adaptation.query_text_views.view_rows import (
    USB_MULTIVIEW_BUILDER_NAME,
    USB_WEAK_BUILDER_NAME,
    QuerySslBacktranslationPair,
    attach_usb_multiview_candidate_pair,
    rows_have_usb_multiview_candidates,
    validate_usb_multiview_candidate_rows,
    validate_usb_weak_rows,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
    load_labeled_query_rows,
)

QUERY_SSL_AUGMENTER_CACHE_SCHEMA_VERSION = "query_ssl_augmenter_cache.v1"
QUERY_SSL_AUGMENTER_SUMMARY_SCHEMA_VERSION = "query_ssl_augmenter_summary.v1"
PRECOMPUTED_USB_CANDIDATES_AUGMENTER = "precomputed_usb_candidates"
NLLB_BACKTRANSLATION_AUGMENTER = "nllb_backtranslation"

QuerySslCandidatePairBuilder = Callable[
    [Sequence[str]],
    Sequence[QuerySslBacktranslationPair],
]
QuerySslPreparationEventSink = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class QuerySslAugmenterSettings:
    """Query SSL strong view source 설정의 typed shape."""

    name: str
    augmenter_type: str
    source_lang: str | None = None
    pivot_languages: tuple[str, ...] = ()
    model_id: str | None = None
    revision: str | None = None
    device: str | None = None
    local_files_only: bool | None = None
    batch_size: int | None = None
    max_new_tokens: int | None = None
    torch_dtype: str | None = None
    cache_dir: str | None = None
    candidate_pair_builder_path: str | None = None


@dataclass(slots=True)
class PreparedQuerySslUnlabeledRows:
    """Query SSL objective가 실제로 소비할 unlabeled row와 cache metadata."""

    rows: list[LabeledQueryRow]
    mode: str
    cache_hit: bool
    uses_strong_view_candidates: bool
    prepared_jsonl_path: Path | None = None
    manifest_path: Path | None = None
    summary_path: Path | None = None

    def build_run_manifest(self) -> dict[str, object]:
        return {
            "query_ssl_augmenter_preparation": {
                "mode": self.mode,
                "cache_hit": self.cache_hit,
                "prepared_jsonl": None
                if self.prepared_jsonl_path is None
                else str(self.prepared_jsonl_path),
                "prepared_manifest_json": None
                if self.manifest_path is None
                else str(self.manifest_path),
                "prepared_summary_json": None
                if self.summary_path is None
                else str(self.summary_path),
            }
        }


@dataclass(frozen=True, slots=True)
class _AugmentationCacheArtifacts:
    jsonl_path: Path
    manifest_path: Path
    summary_path: Path
    cache_key: str


def build_query_ssl_augmenter_manifest(
    settings: QuerySslAugmenterSettings | None,
) -> dict[str, object]:
    """query_ssl_augmenter 설정을 run manifest에 남길 canonical shape로 만든다."""

    if settings is None:
        return {}

    manifest: dict[str, object] = {
        "preset_name": settings.name,
        "augmenter_type": settings.augmenter_type,
    }
    _put_optional_manifest_value(manifest, "source_lang", settings.source_lang)
    if settings.pivot_languages:
        manifest["pivot_languages"] = list(settings.pivot_languages)
    _put_optional_manifest_value(manifest, "model_id", settings.model_id)
    _put_optional_manifest_value(manifest, "revision", settings.revision)
    _put_optional_manifest_value(manifest, "device", settings.device)
    if settings.local_files_only is not None:
        manifest["local_files_only"] = settings.local_files_only
    if settings.batch_size is not None:
        manifest["batch_size"] = settings.batch_size
    if settings.max_new_tokens is not None:
        manifest["max_new_tokens"] = settings.max_new_tokens
    _put_optional_manifest_value(manifest, "torch_dtype", settings.torch_dtype)
    manifest["cache_dir"] = _normalize_optional_text(settings.cache_dir)
    return manifest


def prepare_query_ssl_unlabeled_rows(
    *,
    view_builder_name: str,
    algorithm_name: str,
    rows: Sequence[LabeledQueryRow],
    source_jsonl: str | Path | None,
    augmenter_settings: QuerySslAugmenterSettings | None = None,
    candidate_pair_builder: QuerySslCandidatePairBuilder | None = None,
    event_sink: QuerySslPreparationEventSink | None = None,
) -> PreparedQuerySslUnlabeledRows:
    """algorithm view surface에 맞춰 중앙 unlabeled rows를 준비한다."""

    if view_builder_name == USB_MULTIVIEW_BUILDER_NAME:
        return _prepare_usb_multiview_unlabeled_rows(
            rows=rows,
            source_jsonl=source_jsonl,
            algorithm_name=algorithm_name,
            augmenter_settings=augmenter_settings,
            candidate_pair_builder=candidate_pair_builder,
            event_sink=event_sink,
        )
    if view_builder_name == USB_WEAK_BUILDER_NAME:
        return _prepare_usb_weak_unlabeled_rows(
            rows=rows,
            algorithm_name=algorithm_name,
        )
    raise ValueError(f"Unsupported Query SSL view builder: {view_builder_name}.")


def _prepare_usb_multiview_unlabeled_rows(
    *,
    rows: Sequence[LabeledQueryRow],
    source_jsonl: str | Path | None,
    algorithm_name: str,
    augmenter_settings: QuerySslAugmenterSettings | None,
    candidate_pair_builder: QuerySslCandidatePairBuilder | None,
    event_sink: QuerySslPreparationEventSink | None,
) -> PreparedQuerySslUnlabeledRows:
    """Strict USB형 multiview unlabeled row를 보장한다."""

    effective_rows = list(rows)
    if not effective_rows:
        raise ValueError(f"{algorithm_name} unlabeled_rows must not be empty.")
    if rows_have_usb_multiview_candidates(effective_rows):
        return PreparedQuerySslUnlabeledRows(
            rows=effective_rows,
            mode=PRECOMPUTED_USB_CANDIDATES_AUGMENTER,
            cache_hit=False,
            uses_strong_view_candidates=True,
        )

    if augmenter_settings is None:
        raise ValueError(
            f"{algorithm_name} requires query_ssl_augmenter settings when "
            "unlabeled rows do not include both aug_0 and aug_1."
        )
    if augmenter_settings.augmenter_type == PRECOMPUTED_USB_CANDIDATES_AUGMENTER:
        raise ValueError(
            f"{algorithm_name} requires each unlabeled row to include both aug_0 "
            "and aug_1 when query_ssl_augmenter is precomputed-only."
        )
    if augmenter_settings.augmenter_type != NLLB_BACKTRANSLATION_AUGMENTER:
        raise ValueError(
            "Unsupported query_ssl_augmenter.augmenter_type: "
            f"{augmenter_settings.augmenter_type}"
        )
    if candidate_pair_builder is None:
        raise ValueError(
            f"{algorithm_name} requires a candidate_pair_builder for "
            f"{NLLB_BACKTRANSLATION_AUGMENTER} augmentation."
        )

    cache_artifacts = _resolve_cache_artifacts(
        algorithm_name=algorithm_name,
        augmenter_settings=augmenter_settings,
        rows=effective_rows,
        source_jsonl=source_jsonl,
    )
    if (
        cache_artifacts is not None
        and cache_artifacts.jsonl_path.exists()
        and cache_artifacts.manifest_path.exists()
        and cache_artifacts.summary_path.exists()
    ):
        cached_rows = load_labeled_query_rows(cache_artifacts.jsonl_path)
        validate_usb_multiview_candidate_rows(
            cached_rows,
            context=algorithm_name,
        )
        _emit(
            event_sink,
            "query_ssl_augmenter=cache_hit "
            f"rows={len(cached_rows)} "
            f"prepared_jsonl={cache_artifacts.jsonl_path}",
        )
        return PreparedQuerySslUnlabeledRows(
            rows=cached_rows,
            mode="cache_hit",
            cache_hit=True,
            uses_strong_view_candidates=True,
            prepared_jsonl_path=cache_artifacts.jsonl_path,
            manifest_path=cache_artifacts.manifest_path,
            summary_path=cache_artifacts.summary_path,
        )

    _emit(
        event_sink,
        "query_ssl_augmenter=generate_strict_usb_candidates "
        f"rows={len(effective_rows)} "
        f"batch_size={_format_optional_value(augmenter_settings.batch_size)} "
        f"model_id={_format_optional_value(augmenter_settings.model_id)} "
        f"torch_dtype={_format_optional_value(augmenter_settings.torch_dtype)} "
        f"pivots={list(augmenter_settings.pivot_languages)}",
    )
    candidate_pairs = list(
        candidate_pair_builder([str(row["text"]) for row in effective_rows])
    )
    if len(candidate_pairs) != len(effective_rows):
        raise ValueError(
            "Backtranslation augmenter returned a candidate count that does not "
            "match the source row count."
        )

    prepared_rows: list[LabeledQueryRow] = []
    for row, candidate_pair in zip(effective_rows, candidate_pairs):
        prepared_rows.append(attach_usb_multiview_candidate_pair(row, candidate_pair))

    validate_usb_multiview_candidate_rows(
        prepared_rows,
        context=algorithm_name,
    )

    if cache_artifacts is None:
        _emit(
            event_sink,
            f"query_ssl_augmenter=generated_in_memory rows={len(prepared_rows)}",
        )
        return PreparedQuerySslUnlabeledRows(
            rows=prepared_rows,
            mode="generated_in_memory",
            cache_hit=False,
            uses_strong_view_candidates=True,
        )

    _write_cache_artifacts(
        cache_artifacts=cache_artifacts,
        rows=prepared_rows,
        source_jsonl=source_jsonl,
        query_ssl_augmenter_manifest=build_query_ssl_augmenter_manifest(
            augmenter_settings
        ),
    )
    _emit(
        event_sink,
        "query_ssl_augmenter=generated_and_cached "
        f"rows={len(prepared_rows)} "
        f"prepared_jsonl={cache_artifacts.jsonl_path}",
    )
    return PreparedQuerySslUnlabeledRows(
        rows=prepared_rows,
        mode="generated_and_cached",
        cache_hit=False,
        uses_strong_view_candidates=True,
        prepared_jsonl_path=cache_artifacts.jsonl_path,
        manifest_path=cache_artifacts.manifest_path,
        summary_path=cache_artifacts.summary_path,
    )


def _prepare_usb_weak_unlabeled_rows(
    *,
    rows: Sequence[LabeledQueryRow],
    algorithm_name: str,
) -> PreparedQuerySslUnlabeledRows:
    """USB PseudoLabel처럼 원문 weak view만 필요한 unlabeled row를 검증한다."""

    effective_rows = list(rows)
    if not effective_rows:
        raise ValueError(f"{algorithm_name} unlabeled_rows must not be empty.")
    validate_usb_weak_rows(effective_rows, context=algorithm_name)
    return PreparedQuerySslUnlabeledRows(
        rows=effective_rows,
        mode="raw_weak_text",
        cache_hit=False,
        uses_strong_view_candidates=False,
    )


def _resolve_cache_artifacts(
    *,
    algorithm_name: str,
    augmenter_settings: QuerySslAugmenterSettings,
    rows: Sequence[LabeledQueryRow],
    source_jsonl: str | Path | None,
) -> _AugmentationCacheArtifacts | None:
    cache_dir = _normalize_optional_text(augmenter_settings.cache_dir)
    if cache_dir is None:
        return None

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    augmenter_manifest = build_query_ssl_augmenter_manifest(augmenter_settings)
    cache_key = _build_cache_key(
        rows=rows,
        source_jsonl=source_jsonl,
        augmenter_manifest=augmenter_manifest,
    )
    cache_stem = (
        f"query_ssl_{_slugify(algorithm_name)}_"
        f"{_slugify(augmenter_settings.name)}_{cache_key}"
    )
    jsonl_path = cache_root / f"{cache_stem}.jsonl"
    manifest_path = cache_root / f"{cache_stem}.manifest.json"
    summary_path = cache_root / f"{cache_stem}.summary.json"
    return _AugmentationCacheArtifacts(
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        cache_key=cache_key,
    )


def _build_cache_key(
    *,
    rows: Sequence[LabeledQueryRow],
    source_jsonl: str | Path | None,
    augmenter_manifest: Mapping[str, object],
) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["query_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["text"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("locale", "")).encode("utf-8"))
        digest.update(b"\n")

    cache_payload = {
        "source_jsonl": None if source_jsonl is None else str(source_jsonl),
        "rows_fingerprint": digest.hexdigest(),
        "augmenter_manifest": dict(augmenter_manifest),
    }
    return hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _write_cache_artifacts(
    *,
    cache_artifacts: _AugmentationCacheArtifacts,
    rows: Sequence[LabeledQueryRow],
    source_jsonl: str | Path | None,
    query_ssl_augmenter_manifest: dict[str, object],
) -> None:
    dump_labeled_query_rows(cache_artifacts.jsonl_path, rows)
    generated_at = datetime.now(timezone.utc)
    cache_artifacts.manifest_path.write_text(
        json.dumps(
            {
                "schema_version": QUERY_SSL_AUGMENTER_CACHE_SCHEMA_VERSION,
                "generated_at": generated_at.isoformat(),
                "source_jsonl": None if source_jsonl is None else str(source_jsonl),
                "row_count": len(rows),
                "cache_key": cache_artifacts.cache_key,
                "query_ssl_augmenter": query_ssl_augmenter_manifest,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    cache_artifacts.summary_path.write_text(
        json.dumps(
            _build_cache_summary(
                rows=rows,
                generated_at=generated_at,
                query_ssl_augmenter_manifest=query_ssl_augmenter_manifest,
            ),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_cache_summary(
    *,
    rows: Sequence[LabeledQueryRow],
    generated_at: datetime,
    query_ssl_augmenter_manifest: dict[str, object],
) -> dict[str, object]:
    label_counts: Counter[str] = Counter()
    locale_counts: Counter[str] = Counter()
    aug_0_pivot_lang_counts: Counter[str] = Counter()
    aug_1_pivot_lang_counts: Counter[str] = Counter()
    for row in rows:
        label_counts[str(row["mapped_label_4"])] += 1
        locale_counts[str(row["locale"])] += 1
        aug_0_pivot_lang_counts[str(row.get("aug_0_pivot_lang", ""))] += 1
        aug_1_pivot_lang_counts[str(row.get("aug_1_pivot_lang", ""))] += 1

    return {
        "schema_version": QUERY_SSL_AUGMENTER_SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "row_count": len(rows),
        "query_ssl_augmenter": query_ssl_augmenter_manifest,
        "label_counts": dict(sorted(label_counts.items())),
        "locale_counts": dict(sorted(locale_counts.items())),
        "aug_0_pivot_lang_counts": dict(sorted(aug_0_pivot_lang_counts.items())),
        "aug_1_pivot_lang_counts": dict(sorted(aug_1_pivot_lang_counts.items())),
    }


def _put_optional_manifest_value(
    manifest: dict[str, object],
    key: str,
    value: str | None,
) -> None:
    normalized_value = _normalize_optional_text(value)
    if normalized_value is not None:
        manifest[key] = normalized_value


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _format_optional_value(value: object | None) -> object:
    return "unset" if value is None else value


def _emit(event_sink: QuerySslPreparationEventSink | None, message: str) -> None:
    if event_sink is not None:
        event_sink(message)


def _slugify(value: str) -> str:
    characters = [
        character.lower() if character.isalnum() else "_" for character in value.strip()
    ]
    slug = "".join(characters).strip("_")
    return slug or "query_ssl"

"""Query SSL unlabeled row augmentation preparation and cache."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from methods.adaptation.query_text_views.view_rows import (
    attach_usb_multiview_candidate_pair,
    rows_have_usb_multiview_candidates,
    validate_usb_multiview_candidate_rows,
    validate_usb_weak_rows,
)
from scripts.runtime_adapters.backtranslation_runtime import (
    build_nllb_backtranslation_candidate_pairs,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
    load_labeled_query_rows,
)

QUERY_SSL_AUGMENTER_CACHE_SCHEMA_VERSION = "query_ssl_augmenter_cache.v1"
QUERY_SSL_AUGMENTER_SUMMARY_SCHEMA_VERSION = "query_ssl_augmenter_summary.v1"


@dataclass(slots=True)
class PreparedQuerySslUnlabeledRows:
    """Query SSL objective가 실제로 소비할 unlabeled row와 cache metadata."""

    rows: list[LabeledQueryRow]
    mode: str
    cache_hit: bool
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


def build_query_ssl_augmenter_manifest(cfg) -> dict[str, object]:
    """query_ssl_augmenter Hydra group를 run manifest에 남길 canonical shape."""

    augmenter_cfg = getattr(cfg, "query_ssl_augmenter", None)
    if augmenter_cfg is None:
        return {}

    manifest: dict[str, object] = {
        "preset_name": str(augmenter_cfg.name),
        "augmenter_type": str(augmenter_cfg.augmenter_type),
    }
    if hasattr(augmenter_cfg, "source_lang"):
        manifest["source_lang"] = str(augmenter_cfg.source_lang)
    if hasattr(augmenter_cfg, "pivot_languages"):
        manifest["pivot_languages"] = [
            str(language) for language in augmenter_cfg.pivot_languages
        ]
    if hasattr(augmenter_cfg, "model_id"):
        manifest["model_id"] = str(augmenter_cfg.model_id)
    if hasattr(augmenter_cfg, "revision"):
        manifest["revision"] = str(augmenter_cfg.revision)
    if hasattr(augmenter_cfg, "device"):
        manifest["device"] = str(augmenter_cfg.device)
    if hasattr(augmenter_cfg, "local_files_only"):
        manifest["local_files_only"] = bool(augmenter_cfg.local_files_only)
    if hasattr(augmenter_cfg, "batch_size"):
        manifest["batch_size"] = int(augmenter_cfg.batch_size)
    if hasattr(augmenter_cfg, "max_new_tokens"):
        manifest["max_new_tokens"] = int(augmenter_cfg.max_new_tokens)
    if hasattr(augmenter_cfg, "torch_dtype"):
        manifest["torch_dtype"] = str(augmenter_cfg.torch_dtype)
    if hasattr(augmenter_cfg, "cache_dir"):
        cache_dir = str(augmenter_cfg.cache_dir).strip()
        manifest["cache_dir"] = cache_dir or None
    return manifest


def prepare_usb_multiview_unlabeled_rows(
    cfg,
    *,
    rows: Sequence[LabeledQueryRow],
    source_jsonl: str | Path | None,
    algorithm_name: str,
) -> PreparedQuerySslUnlabeledRows:
    """Strict USB형 multiview unlabeled row를 보장한다."""

    effective_rows = list(rows)
    if not effective_rows:
        raise ValueError(f"{algorithm_name} unlabeled_rows must not be empty.")
    if rows_have_usb_multiview_candidates(effective_rows):
        return PreparedQuerySslUnlabeledRows(
            rows=effective_rows,
            mode="precomputed_usb_candidates",
            cache_hit=False,
        )

    augmenter_type = str(cfg.query_ssl_augmenter.augmenter_type)
    if augmenter_type == "precomputed_usb_candidates":
        raise ValueError(
            f"{algorithm_name} requires each unlabeled row to include both aug_0 "
            "and aug_1 when query_ssl_augmenter is precomputed-only."
        )
    if augmenter_type != "nllb_backtranslation":
        raise ValueError(
            f"Unsupported query_ssl_augmenter.augmenter_type: {augmenter_type}"
        )

    cache_artifacts = _resolve_cache_artifacts(
        cfg=cfg,
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
        print(
            "query_ssl_augmenter=cache_hit "
            f"rows={len(cached_rows)} "
            f"prepared_jsonl={cache_artifacts.jsonl_path}",
            flush=True,
        )
        return PreparedQuerySslUnlabeledRows(
            rows=cached_rows,
            mode="cache_hit",
            cache_hit=True,
            prepared_jsonl_path=cache_artifacts.jsonl_path,
            manifest_path=cache_artifacts.manifest_path,
            summary_path=cache_artifacts.summary_path,
        )

    print(
        "query_ssl_augmenter=generate_strict_usb_candidates "
        f"rows={len(effective_rows)} "
        f"batch_size={int(cfg.query_ssl_augmenter.batch_size)} "
        f"model_id={str(cfg.query_ssl_augmenter.model_id)} "
        f"torch_dtype={str(getattr(cfg.query_ssl_augmenter, 'torch_dtype', 'auto'))} "
        f"pivots={list(cfg.query_ssl_augmenter.pivot_languages)}",
        flush=True,
    )
    candidate_pairs = build_nllb_backtranslation_candidate_pairs(
        cfg, texts=[str(row["text"]) for row in effective_rows]
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
        print(
            f"query_ssl_augmenter=generated_in_memory rows={len(prepared_rows)}",
            flush=True,
        )
        return PreparedQuerySslUnlabeledRows(
            rows=prepared_rows,
            mode="generated_in_memory",
            cache_hit=False,
        )

    _write_cache_artifacts(
        cache_artifacts=cache_artifacts,
        rows=prepared_rows,
        source_jsonl=source_jsonl,
        query_ssl_augmenter_manifest=build_query_ssl_augmenter_manifest(cfg),
    )
    print(
        "query_ssl_augmenter=generated_and_cached "
        f"rows={len(prepared_rows)} "
        f"prepared_jsonl={cache_artifacts.jsonl_path}",
        flush=True,
    )
    return PreparedQuerySslUnlabeledRows(
        rows=prepared_rows,
        mode="generated_and_cached",
        cache_hit=False,
        prepared_jsonl_path=cache_artifacts.jsonl_path,
        manifest_path=cache_artifacts.manifest_path,
        summary_path=cache_artifacts.summary_path,
    )


def prepare_usb_weak_unlabeled_rows(
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
    )


def _resolve_cache_artifacts(
    *,
    cfg,
    rows: Sequence[LabeledQueryRow],
    source_jsonl: str | Path | None,
) -> _AugmentationCacheArtifacts | None:
    cache_dir = str(getattr(cfg.query_ssl_augmenter, "cache_dir", "") or "").strip()
    if not cache_dir:
        return None

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    augmenter_manifest = build_query_ssl_augmenter_manifest(cfg)
    cache_key = _build_cache_key(
        rows=rows,
        source_jsonl=source_jsonl,
        augmenter_manifest=augmenter_manifest,
    )
    cache_stem = (
        f"query_ssl_{_slugify(str(cfg.query_ssl_method.algorithm_name))}_"
        f"{_slugify(str(cfg.query_ssl_augmenter.name))}_{cache_key}"
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
    augmenter_manifest: dict[str, object],
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
        "augmenter_manifest": augmenter_manifest,
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


def _slugify(value: str) -> str:
    characters = [
        character.lower() if character.isalnum() else "_" for character in value.strip()
    ]
    slug = "".join(characters).strip("_")
    return slug or "query_ssl"

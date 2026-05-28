"""중앙 Query SSL unlabeled view 준비 runtime adapter."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path

from methods.adaptation.query_text_views.unlabeled_preparation import (
    PreparedQuerySslUnlabeledRows,
    QuerySslAugmenterSettings,
    QuerySslCandidatePairBuilder,
)
from methods.adaptation.query_text_views.unlabeled_preparation import (
    build_query_ssl_augmenter_manifest as build_manifest_from_settings,
)
from methods.adaptation.query_text_views.unlabeled_preparation import (
    prepare_query_ssl_unlabeled_rows as prepare_methods_query_ssl_unlabeled_rows,
)
from methods.ssl.base import QuerySslAlgorithmDescriptor
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


def prepare_query_ssl_unlabeled_rows(
    *,
    cfg: object,
    descriptor: QuerySslAlgorithmDescriptor,
    rows: list[LabeledQueryRow],
    source_jsonl: str | Path | None,
) -> PreparedQuerySslUnlabeledRows:
    """Hydra cfg와 runtime callable을 methods-owned view preparation으로 넘긴다."""

    augmenter_settings = build_query_ssl_augmenter_settings(cfg)
    return prepare_methods_query_ssl_unlabeled_rows(
        view_builder_name=descriptor.required_views.view_builder_name,
        algorithm_name=descriptor.display_name,
        rows=rows,
        source_jsonl=source_jsonl,
        augmenter_settings=augmenter_settings,
        candidate_pair_builder=_resolve_candidate_pair_builder(
            cfg=cfg,
            settings=augmenter_settings,
        ),
        event_sink=_print_preparation_event,
    )


def build_query_ssl_augmenter_manifest(cfg: object) -> dict[str, object]:
    """query_ssl_augmenter Hydra group를 run manifest shape로 변환한다."""

    return build_manifest_from_settings(
        build_query_ssl_augmenter_settings(cfg)
    )


def build_query_ssl_augmenter_settings(
    cfg: object,
) -> QuerySslAugmenterSettings | None:
    """Hydra query_ssl_augmenter group을 typed settings로 정규화한다."""

    augmenter_cfg = getattr(cfg, "query_ssl_augmenter", None)
    if augmenter_cfg is None:
        return None

    return QuerySslAugmenterSettings(
        name=str(augmenter_cfg.name),
        augmenter_type=str(augmenter_cfg.augmenter_type),
        source_lang=_optional_text(augmenter_cfg, "source_lang"),
        pivot_languages=tuple(
            str(language)
            for language in getattr(augmenter_cfg, "pivot_languages", ()) or ()
        ),
        model_id=_optional_text(augmenter_cfg, "model_id"),
        revision=_optional_text(augmenter_cfg, "revision"),
        device=_optional_text(augmenter_cfg, "device"),
        local_files_only=_optional_bool(augmenter_cfg, "local_files_only"),
        batch_size=_optional_int(augmenter_cfg, "batch_size"),
        max_new_tokens=_optional_int(augmenter_cfg, "max_new_tokens"),
        torch_dtype=_optional_text(augmenter_cfg, "torch_dtype"),
        cache_dir=_optional_text(augmenter_cfg, "cache_dir"),
        candidate_pair_builder_path=_optional_text(
            augmenter_cfg,
            "candidate_pair_builder_path",
        ),
    )


def _resolve_candidate_pair_builder(
    *,
    cfg: object,
    settings: QuerySslAugmenterSettings | None,
) -> QuerySslCandidatePairBuilder | None:
    if settings is None or settings.candidate_pair_builder_path is None:
        return None
    factory = _load_callable(settings.candidate_pair_builder_path)
    return factory(cfg)


def _load_callable(path: str) -> Callable[[object], QuerySslCandidatePairBuilder]:
    module_name, _, function_name = path.strip().rpartition(".")
    if not module_name or not function_name:
        raise ValueError(f"Invalid candidate_pair_builder_path: {path}")
    loaded = getattr(import_module(module_name), function_name)
    if not callable(loaded):
        raise TypeError(f"candidate_pair_builder_path is not callable: {path}")
    return loaded


def _optional_text(cfg: object, key: str) -> str | None:
    if not hasattr(cfg, key):
        return None
    value = getattr(cfg, key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_bool(cfg: object, key: str) -> bool | None:
    if not hasattr(cfg, key):
        return None
    value = getattr(cfg, key)
    if value is None:
        return None
    return bool(value)


def _optional_int(cfg: object, key: str) -> int | None:
    if not hasattr(cfg, key):
        return None
    value = getattr(cfg, key)
    if value is None:
        return None
    return int(value)


def _print_preparation_event(message: str) -> None:
    print(message, flush=True)

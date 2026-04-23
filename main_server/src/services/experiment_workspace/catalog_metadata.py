"""Hydra config 기반 catalog metadata 추출 helper."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from main_server.src.services.experiment_workspace.payloads import (
    CatalogOverrideFieldPayload,
)

CatalogMetadataResolver = Callable[
    [Path, Mapping[str, object], tuple[str, ...] | None],
    dict[str, object],
]
CatalogCoreMethodResolver = Callable[[Path, Mapping[str, object]], str | None]
CatalogTagResolver = Callable[[Path, Mapping[str, object]], tuple[str, ...]]
CatalogMetadataScalar = str | int | float | bool | None


def extract_default_groups(raw: Mapping[str, object]) -> tuple[str, ...]:
    """Hydra defaults를 compiler-friendly group selector로 정규화한다."""

    defaults = raw.get("defaults")
    if not isinstance(defaults, list):
        return ()
    resolved: list[str] = []
    for entry in defaults:
        if isinstance(entry, str):
            normalized_entry = entry.strip()
            if not normalized_entry or normalized_entry == "_self_":
                continue
            if "hydra/" in normalized_entry:
                continue
            resolved.append(normalized_entry.lstrip("/"))
            continue
        if not isinstance(entry, Mapping):
            continue
        for raw_key, raw_value in entry.items():
            key = str(raw_key).strip()
            if not key or "hydra/" in key:
                continue
            normalized_key = key.replace("override /", "").lstrip("/")
            resolved.append(f"{normalized_key}={raw_value}")
    return tuple(resolved)


def declared_fields(raw: Mapping[str, object]) -> tuple[str, ...]:
    """defaults를 제외한 top-level field를 catalog에 노출한다."""

    return tuple(sorted(str(key) for key in raw if str(key) != "defaults"))


def extract_override_fields(
    raw: Mapping[str, object],
) -> tuple[CatalogOverrideFieldPayload, ...]:
    """안전한 top-level scalar field만 override editor surface로 승격한다."""

    override_fields: list[CatalogOverrideFieldPayload] = []
    for key, value in raw.items():
        field_name = str(key)
        if field_name in {"defaults", "name", "algorithm_profile_name"}:
            continue
        override_field = _build_override_field(field_name, value)
        if override_field is None:
            continue
        override_fields.append(override_field)
    return tuple(sorted(override_fields, key=lambda field: field.field_name))


def extract_scalar_metadata(
    raw: Mapping[str, object],
    *,
    metadata_keys: tuple[str, ...] | None = None,
) -> dict[str, CatalogMetadataScalar]:
    """catalog 카드에 바로 보여줄 scalar metadata만 추린다."""

    keys = raw.keys() if metadata_keys is None else metadata_keys
    metadata: dict[str, CatalogMetadataScalar] = {}
    for key in keys:
        string_key = str(key)
        if string_key == "defaults":
            continue
        value = raw.get(string_key)
        if _is_scalar_metadata_value(value):
            metadata[string_key] = value
    return metadata


def build_dataset_preset_metadata(
    _path: Path,
    raw: Mapping[str, object],
    _metadata_keys: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """dataset alias의 asset path, provenance, readiness를 조립한다."""

    asset_paths = {
        "train_jsonl": string_or_none(raw.get("train_jsonl")),
        "validation_jsonl": string_or_none(raw.get("validation_jsonl")),
        "test_jsonl": string_or_none(raw.get("test_jsonl")),
        "prototype_input_jsonl": string_or_none(raw.get("prototype_input_jsonl")),
        "query_dev_jsonl": string_or_none(raw.get("query_dev_jsonl")),
        "query_calibration_jsonl": string_or_none(raw.get("query_calibration_jsonl")),
        "unlabeled_query_pool_jsonl": string_or_none(
            raw.get("unlabeled_query_pool_jsonl")
        ),
    }
    sources = raw.get("sources")
    serialized_sources: dict[str, object] = {}
    if isinstance(sources, Mapping):
        for source_name, source_raw in sources.items():
            if not isinstance(source_raw, Mapping):
                continue
            serialized_sources[str(source_name)] = {
                "kind": string_or_none(source_raw.get("kind")),
                "dataset_id": string_or_none(source_raw.get("dataset_id")),
                "split": string_or_none(source_raw.get("split")),
                "data_file": string_or_none(source_raw.get("data_file")),
                "reference_urls": [
                    str(url)
                    for url in source_raw.get("reference_urls", [])
                    if isinstance(url, str) and url.strip()
                ],
            }
    unlabeled_ready = bool(asset_paths["unlabeled_query_pool_jsonl"])
    return {
        "asset_paths": asset_paths,
        "readiness": {
            "seed_ready": bool(
                asset_paths["train_jsonl"]
                and asset_paths["validation_jsonl"]
                and asset_paths["test_jsonl"]
            ),
            "central_supervised_ready": bool(
                asset_paths["train_jsonl"]
                and asset_paths["validation_jsonl"]
                and asset_paths["test_jsonl"]
            ),
            "central_fixmatch_ready": bool(
                asset_paths["train_jsonl"]
                and asset_paths["validation_jsonl"]
                and asset_paths["test_jsonl"]
                and unlabeled_ready
            ),
            "federated_baseline_ready": bool(
                asset_paths["train_jsonl"] and asset_paths["validation_jsonl"]
            ),
        },
        "query_asset_status": {
            "query_dev_available": bool(asset_paths["query_dev_jsonl"]),
            "query_calibration_available": bool(
                asset_paths["query_calibration_jsonl"]
            ),
            "unlabeled_query_pool_available": unlabeled_ready,
        },
        "sources": serialized_sources,
    }


def build_federated_run_preset_metadata(
    _path: Path,
    raw: Mapping[str, object],
    metadata_keys: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """federated run preset에 participant count 의미를 명시적으로 붙인다."""

    metadata = extract_scalar_metadata(raw, metadata_keys=metadata_keys)
    metadata["count_semantics"] = {
        "client_count": "simulation_participants",
        "live_agent_roster": "not_included",
        "round_selected_agents": "not_included",
    }
    return metadata


def resolve_catalog_item_name(
    raw: Mapping[str, object],
    *,
    fallback: str | None = None,
) -> str:
    """Hydra preset의 canonical display/item name을 해석한다."""

    for key in ("name", "algorithm_profile_name"):
        value = string_or_none(raw.get(key))
        if value is not None:
            return value
    if fallback is None:
        raise ValueError("Catalog item name is missing and no fallback was given.")
    return fallback


def string_or_none(value: object) -> str | None:
    """값이 존재하면 문자열로 정규화하고, 없으면 None을 유지한다."""

    if value is None:
        return None
    return str(value)


def _build_override_field(
    field_name: str,
    value: object,
) -> CatalogOverrideFieldPayload | None:
    if isinstance(value, bool):
        return CatalogOverrideFieldPayload(
            field_name=field_name,
            value_kind="boolean",
            default_value=value,
        )
    if isinstance(value, int):
        return CatalogOverrideFieldPayload(
            field_name=field_name,
            value_kind="integer",
            default_value=value,
        )
    if isinstance(value, float):
        return CatalogOverrideFieldPayload(
            field_name=field_name,
            value_kind="number",
            default_value=value,
        )
    if isinstance(value, str):
        return CatalogOverrideFieldPayload(
            field_name=field_name,
            value_kind="string",
            default_value=value,
        )
    return None


def _is_scalar_metadata_value(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))

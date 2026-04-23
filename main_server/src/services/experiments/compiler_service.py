"""Workspace manifest를 Hydra/script preview로 번역하는 compiler service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from main_server.src.services.experiments.catalog_service import (
    ExperimentCatalogService,
)
from main_server.src.services.experiments.payloads import (
    CatalogItemPayload,
    CatalogSectionPayload,
    CatalogTrackPayload,
)
from shared.src.contracts.workspace_manifest_contracts import (
    ResolvedExperimentPlanPayload,
    ResolvedWorkspaceSelectionPayload,
    WorkspaceConfigScalar,
    WorkspaceManifestPayload,
    WorkspaceSelectionPayload,
)


@dataclass(slots=True)
class ExperimentCompilerService:
    """Workspace manifest를 read-only 실행 preview로 compile한다."""

    catalog_service: ExperimentCatalogService

    def compile_manifest(
        self,
        manifest: WorkspaceManifestPayload,
    ) -> ResolvedExperimentPlanPayload:
        """WorkspaceManifest를 기존 Hydra/script 표면으로 번역한다."""

        catalog = self.catalog_service.build_catalog()
        track = _find_track(catalog.tracks, manifest.track_name)
        entrypoints_section = _find_section(track.sections, "entrypoints")
        entrypoint_item = _find_variant_item(
            entrypoints_section.items,
            manifest.entrypoint_name,
        )
        if entrypoint_item.compile_support != "entrypoint":
            raise ValueError(
                "Workspace entrypoint is not compileable: "
                f"{manifest.entrypoint_name}."
            )
        if entrypoint_item.script_path is None:
            raise ValueError(
                "Catalog entrypoint is missing script_path: "
                f"{manifest.entrypoint_name}."
            )

        seen_slots: set[str] = set()
        selection_default_groups: list[str] = []
        hydra_overrides: list[str] = []
        resolved_selections: list[ResolvedWorkspaceSelectionPayload] = []
        warnings: list[str] = []

        for selection in manifest.selections:
            if selection.slot_name in seen_slots:
                raise ValueError(
                    f"Duplicate workspace slot is not allowed: {selection.slot_name}."
                )
            seen_slots.add(selection.slot_name)

            section = _find_section(track.sections, selection.section_name)
            item = _find_variant_item(
                section.items,
                selection.variant_profile_name,
            )
            _validate_selection_against_item(selection, item)

            if item.compile_support != "preset_selector":
                detail = item.compile_blocker_reason or (
                    "선택한 catalog item은 아직 compile 규칙이 없다."
                )
                raise ValueError(
                    "Workspace selection is not compileable yet: "
                    f"{selection.section_name}/{selection.variant_profile_name}. "
                    f"{detail}"
                )
            if item.preset_group is None:
                raise ValueError(
                    "Preset-selector catalog item is missing preset_group: "
                    f"{selection.section_name}/{selection.variant_profile_name}."
                )

            compiled_selector = (
                f"{item.preset_group}={item.variant_profile_name or item.item_name}"
            )
            compiled_overrides: list[str] = []
            selection_default_groups.append(compiled_selector)
            for key, value in selection.override_patch.items():
                compiled_override = (
                    f"{item.preset_group}.{key}={_format_hydra_value(value)}"
                )
                compiled_overrides.append(compiled_override)
                hydra_overrides.append(compiled_override)

            resolved_selections.append(
                ResolvedWorkspaceSelectionPayload(
                    slot_name=selection.slot_name,
                    section_name=selection.section_name,
                    family_name=item.family_name,
                    core_method_name=item.core_method_name,
                    variant_profile_name=(
                        item.variant_profile_name or selection.variant_profile_name
                    ),
                    source_of_truth=item.source_of_truth,
                    preset_group=item.preset_group,
                    compiled_selector=compiled_selector,
                    compiled_overrides=tuple(compiled_overrides),
                )
            )

        for key, value in manifest.global_override_patch.items():
            hydra_overrides.append(f"{key}={_format_hydra_value(value)}")

        effective_groups = _merge_group_assignments(
            entrypoint_item.default_groups,
            tuple(selection_default_groups),
        )
        hydra_override_map = _parse_hydra_override_map(tuple(hydra_overrides))
        warnings.extend(
            self._collect_compile_warnings(
                manifest=manifest,
                entrypoint_item=entrypoint_item,
                effective_groups=effective_groups,
                hydra_override_map=hydra_override_map,
            )
        )
        self._validate_compile_requirements(
            manifest=manifest,
            effective_groups=effective_groups,
            hydra_override_map=hydra_override_map,
        )

        script_path = entrypoint_item.script_path
        command_args = (
            "uv",
            "run",
            "python",
            script_path,
            *selection_default_groups,
            *hydra_overrides,
        )
        return ResolvedExperimentPlanPayload(
            manifest_id=manifest.manifest_id,
            track_name=manifest.track_name,
            entrypoint_name=manifest.entrypoint_name,
            job_config_path=entrypoint_item.source_of_truth,
            script_path=script_path,
            base_default_groups=entrypoint_item.default_groups,
            selection_default_groups=tuple(selection_default_groups),
            hydra_overrides=tuple(hydra_overrides),
            command_args=command_args,
            resolved_selections=tuple(resolved_selections),
            warnings=tuple(warnings),
        )

    def _collect_compile_warnings(
        self,
        *,
        manifest: WorkspaceManifestPayload,
        entrypoint_item: CatalogItemPayload,
        effective_groups: dict[str, str],
        hydra_override_map: dict[str, str],
    ) -> list[str]:
        warnings: list[str] = []
        if manifest.entrypoint_name != "run_federated_simulation":
            return warnings

        warnings.append(
            "run_federated_simulation의 client_count는 live agent roster가 아니라 "
            "synthetic simulation participant 수를 뜻한다."
        )
        if _resolve_effective_string_override(
            hydra_override_map,
            "shard_policy.name",
        ) not in (None, "label_dominant"):
            return warnings

        dataset_name = effective_groups.get("dataset")
        federated_run_preset = effective_groups.get("federated_run_preset")
        if dataset_name is None or federated_run_preset is None:
            return warnings

        dataset_raw = self.catalog_service.load_config_group_item(
            relative_dir="scripts/conf/dataset",
            item_name=dataset_name,
        )
        label_count = len(
            [
                category
                for category in dataset_raw.get("prototype_expected_categories", [])
                if isinstance(category, str) and category.strip()
            ]
        )
        if label_count <= 0:
            return warnings

        preset_raw = self.catalog_service.load_config_group_item(
            relative_dir="scripts/conf/federated_run_preset",
            item_name=federated_run_preset,
        )
        client_count = _resolve_effective_int_override(
            hydra_override_map,
            key="federated_run_preset.client_count",
            fallback=preset_raw.get("client_count"),
        )
        if client_count is None:
            return warnings
        if client_count > label_count + 1:
            warnings.append(
                "현재 label_dominant shard policy에서는 "
                f"client_count={client_count}, label_count={label_count} 조합에서 "
                "빈 shard 또는 거의 비어 있는 shard가 생길 수 있다."
            )
        return warnings

    def _validate_compile_requirements(
        self,
        *,
        manifest: WorkspaceManifestPayload,
        effective_groups: dict[str, str],
        hydra_override_map: dict[str, str],
    ) -> None:
        if manifest.entrypoint_name != "train_lora_fixmatch":
            return
        if _has_non_null_override(
            hydra_override_map,
            "unlabeled_jsonl",
            "query_ssl_train_source.unlabeled_jsonl",
        ):
            return

        train_source_name = effective_groups.get("query_ssl_train_source")
        if train_source_name is None:
            raise ValueError(
                "FixMatch compile readiness check failed: "
                "query_ssl_train_source preset을 결정할 수 없습니다."
            )
        train_source_raw = self.catalog_service.load_config_group_item(
            relative_dir="scripts/conf/query_ssl_train_source",
            item_name=train_source_name,
        )
        source_unlabeled = _string_or_none(train_source_raw.get("unlabeled_jsonl"))
        if source_unlabeled is None or source_unlabeled == "null":
            raise ValueError(
                "FixMatch compile readiness check failed: "
                f"query_ssl_train_source={train_source_name} 에 "
                "unlabeled_jsonl이 없습니다."
            )
        if source_unlabeled != "${dataset.unlabeled_query_pool_jsonl}":
            return

        dataset_name = effective_groups.get("dataset")
        if dataset_name is None:
            raise ValueError(
                "FixMatch compile readiness check failed: "
                "dataset preset을 결정할 수 없습니다."
            )
        dataset_raw = self.catalog_service.load_config_group_item(
            relative_dir="scripts/conf/dataset",
            item_name=dataset_name,
        )
        dataset_unlabeled = _string_or_none(
            dataset_raw.get("unlabeled_query_pool_jsonl")
        )
        if dataset_unlabeled is None or dataset_unlabeled == "null":
            raise ValueError(
                "FixMatch compile readiness check failed: "
                f"dataset={dataset_name} 는 "
                "unlabeled_query_pool_jsonl이 아직 비어 있습니다. "
                "다른 query_ssl_train_source preset을 고르거나 "
                "unlabeled_jsonl override를 직접 제공하세요."
            )


def _find_track(
    tracks: tuple[CatalogTrackPayload, ...],
    track_name: str,
) -> CatalogTrackPayload:
    for track in tracks:
        if track.track_name == track_name:
            return track
    raise ValueError(f"Unsupported workspace track: {track_name}.")


def _find_section(
    sections: tuple[CatalogSectionPayload, ...],
    section_name: str,
) -> CatalogSectionPayload:
    for section in sections:
        if section.section_name == section_name:
            return section
    raise ValueError(f"Unsupported catalog section: {section_name}.")


def _find_variant_item(
    items: tuple[CatalogItemPayload, ...],
    variant_profile_name: str,
) -> CatalogItemPayload:
    for item in items:
        if item.variant_profile_name == variant_profile_name:
            return item
        if item.item_name == variant_profile_name:
            return item
    raise ValueError(f"Unsupported catalog item: {variant_profile_name}.")


def _validate_selection_against_item(
    selection: WorkspaceSelectionPayload,
    item: CatalogItemPayload,
) -> None:
    if (
        selection.family_name is not None
        and item.family_name is not None
        and selection.family_name != item.family_name
    ):
        raise ValueError(
            "Workspace selection family mismatch: "
            f"expected={selection.family_name}, actual={item.family_name}."
        )
    if (
        selection.core_method_name is not None
        and item.core_method_name is not None
        and selection.core_method_name != item.core_method_name
    ):
        raise ValueError(
            "Workspace selection core method mismatch: "
            f"expected={selection.core_method_name}, actual={item.core_method_name}."
        )


def _format_hydra_value(value: WorkspaceConfigScalar) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _merge_group_assignments(*group_sets: tuple[str, ...]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for group_set in group_sets:
        for entry in group_set:
            if "=" not in entry:
                continue
            key, value = entry.split("=", 1)
            merged[key] = value
    return merged


def _parse_hydra_override_map(overrides: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        parsed[key] = value
    return parsed


def _has_non_null_override(
    hydra_override_map: dict[str, str],
    *keys: str,
) -> bool:
    return any(
        _resolve_effective_string_override(hydra_override_map, key)
        not in (None, "null")
        for key in keys
    )


def _resolve_effective_string_override(
    hydra_override_map: dict[str, str],
    key: str,
) -> str | None:
    value = hydra_override_map.get(key)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _resolve_effective_int_override(
    hydra_override_map: dict[str, str],
    *,
    key: str,
    fallback: Any,
) -> int | None:
    raw_value = _resolve_effective_string_override(hydra_override_map, key)
    if raw_value is None:
        raw_value = _string_or_none(fallback)
    if raw_value is None or raw_value == "null":
        return None
    return int(raw_value)


def _string_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None

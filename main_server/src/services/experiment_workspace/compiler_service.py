"""Workspace manifest를 Hydra/script preview로 번역하는 compiler service."""

from __future__ import annotations

from dataclasses import dataclass, field

from main_server.src.services.experiment_workspace.catalog_service import (
    ExperimentCatalogService,
)
from main_server.src.services.experiment_workspace.compile_policies import (
    DEFAULT_EXPERIMENT_COMPILE_POLICY_REGISTRY,
    ExperimentCompileContext,
    ExperimentCompilePolicyRegistry,
)
from main_server.src.services.experiment_workspace.payloads import (
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
    compile_policy_registry: ExperimentCompilePolicyRegistry = field(
        default_factory=lambda: DEFAULT_EXPERIMENT_COMPILE_POLICY_REGISTRY
    )

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
        compile_context = ExperimentCompileContext(
            manifest=manifest,
            entrypoint_item=entrypoint_item,
            effective_groups=effective_groups,
            hydra_override_map=hydra_override_map,
            catalog_service=self.catalog_service,
        )
        compile_policy = self.compile_policy_registry.resolve(
            entrypoint_item.item_name
        )
        warnings.extend(
            compile_policy.collect_warnings(context=compile_context)
        )
        compile_policy.validate_requirements(context=compile_context)

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

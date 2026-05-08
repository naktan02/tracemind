"""Workspace selection을 Hydra selector/override payload로 compile한다."""

from __future__ import annotations

from dataclasses import dataclass

from main_server.src.services.experiment_workspace.compiler.catalog_lookup import (
    find_section,
    find_variant_item,
)
from main_server.src.services.experiment_workspace.compiler.hydra_overrides import (
    build_preset_hydra_override,
)
from main_server.src.services.experiment_workspace.payloads import (
    CatalogItemPayload,
    CatalogTrackPayload,
)
from shared.src.contracts.workspace_manifest_contracts import (
    ResolvedWorkspaceSelectionPayload,
    WorkspaceSelectionPayload,
)


@dataclass(frozen=True, slots=True)
class CompiledWorkspaceSelections:
    """Workspace selection compile 결과."""

    selection_default_groups: tuple[str, ...]
    hydra_overrides: tuple[str, ...]
    resolved_selections: tuple[ResolvedWorkspaceSelectionPayload, ...]


def compile_workspace_selections(
    *,
    track: CatalogTrackPayload,
    selections: tuple[WorkspaceSelectionPayload, ...],
) -> CompiledWorkspaceSelections:
    seen_slots: set[str] = set()
    selection_default_groups: list[str] = []
    hydra_overrides: list[str] = []
    resolved_selections: list[ResolvedWorkspaceSelectionPayload] = []

    for selection in selections:
        if selection.slot_name in seen_slots:
            raise ValueError(
                f"Duplicate workspace slot is not allowed: {selection.slot_name}."
            )
        seen_slots.add(selection.slot_name)

        section = find_section(track.sections, selection.section_name)
        item = find_variant_item(
            section.items,
            selection.variant_profile_name,
        )
        _validate_selection_against_item(selection, item)
        preset_group = _require_compileable_preset_selector(selection, item)

        selector_name = (
            item.compiled_selector_name or item.variant_profile_name or item.item_name
        )
        selector_group = _selector_group_for_item(item)
        compiled_selector = f"{selector_group}={selector_name}"
        compiled_overrides: list[str] = []
        selection_default_groups.append(compiled_selector)
        effective_override_patch = {
            **item.default_override_patch,
            **selection.override_patch,
        }
        for key, value in effective_override_patch.items():
            compiled_override = build_preset_hydra_override(
                preset_group=preset_group,
                key=key,
                value=value,
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

    return CompiledWorkspaceSelections(
        selection_default_groups=tuple(selection_default_groups),
        hydra_overrides=tuple(hydra_overrides),
        resolved_selections=tuple(resolved_selections),
    )


def _require_compileable_preset_selector(
    selection: WorkspaceSelectionPayload,
    item: CatalogItemPayload,
) -> str:
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
    return item.preset_group


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


def _selector_group_for_item(item: CatalogItemPayload) -> str:
    raw_selector_group = item.metadata.get("selector_group")
    if isinstance(raw_selector_group, str) and raw_selector_group.strip():
        return raw_selector_group.strip()
    if item.preset_group is None:
        raise ValueError(f"Catalog item is missing preset_group: {item.item_name}.")
    return item.preset_group

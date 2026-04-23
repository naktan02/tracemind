"""Catalog section/item builder helper."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from main_server.src.services.experiment_workspace.catalog_constants import (
    PHASE2_METADATA_ONLY_BLOCKER,
)
from main_server.src.services.experiment_workspace.catalog_metadata import (
    CatalogCoreMethodResolver,
    CatalogMetadataResolver,
    CatalogTagResolver,
    declared_fields,
    extract_default_groups,
    extract_override_fields,
    extract_scalar_metadata,
    resolve_catalog_item_name,
)
from main_server.src.services.experiment_workspace.payloads import (
    CatalogItemPayload,
    CatalogSectionPayload,
)
from shared.src.config.adapter_family_metadata import (
    SharedAdapterFamilyMetadata,
)
from shared.src.config.registry_catalog_metadata import RegistryCatalogEntry

LoadYamlMapping = Callable[[Path], dict[str, object]]
IterYamlFiles = Callable[[str], tuple[Path, ...]]
RelativeRepoPath = Callable[[Path], str]
ResolveScriptPath = Callable[[str], str]
SourceOfTruthForModule = Callable[[str], str]
RuntimePathResolver = Callable[[RegistryCatalogEntry], tuple[str, ...]]


def build_entrypoint_section(
    *,
    section_name: str,
    display_name: str,
    description: str,
    relative_paths: tuple[str, ...],
    supported_runtime_paths: tuple[str, ...],
    repo_root: Path,
    load_yaml_mapping: LoadYamlMapping,
    relative_repo_path: RelativeRepoPath,
    resolve_script_path: ResolveScriptPath,
) -> CatalogSectionPayload:
    """Hydra job config 목록을 entrypoint section으로 직렬화한다."""

    items: list[CatalogItemPayload] = []
    for relative_path in relative_paths:
        path = repo_root / relative_path
        raw = load_yaml_mapping(path)
        item_name = path.stem
        source_of_truth = relative_repo_path(path)
        items.append(
            CatalogItemPayload(
                item_name=item_name,
                display_name=item_name,
                item_kind="experiment_entrypoint",
                family_name=section_name,
                core_method_name=item_name,
                variant_profile_name=item_name,
                source_of_truth=source_of_truth,
                source_kind="hydra_job_config",
                compile_support="entrypoint",
                script_path=resolve_script_path(source_of_truth),
                supported_runtime_paths=supported_runtime_paths,
                default_groups=extract_default_groups(raw),
                declared_fields=declared_fields(raw),
                override_fields=extract_override_fields(raw),
                metadata=extract_scalar_metadata(raw),
            )
        )
    return CatalogSectionPayload(
        section_name=section_name,
        display_name=display_name,
        item_kind="experiment_entrypoint",
        description=description,
        source_of_truth="scripts/conf",
        source_kind="hydra_job_config",
        selection_mode="single_required",
        items=tuple(items),
    )


def build_config_group_section(
    *,
    section_name: str,
    display_name: str,
    description: str,
    relative_dir: str,
    item_kind: str,
    family_name: str,
    preset_group: str,
    supported_runtime_paths: tuple[str, ...],
    iter_yaml_files: IterYamlFiles,
    load_yaml_mapping: LoadYamlMapping,
    relative_repo_path: RelativeRepoPath,
    core_method_resolver: CatalogCoreMethodResolver | None = None,
    metadata_keys: tuple[str, ...] | None = None,
    tag_resolver: CatalogTagResolver | None = None,
    metadata_resolver: CatalogMetadataResolver | None = None,
) -> CatalogSectionPayload:
    """Hydra config group 디렉터리를 preset section으로 직렬화한다."""

    items: list[CatalogItemPayload] = []
    for path in iter_yaml_files(relative_dir):
        raw = load_yaml_mapping(path)
        item_name = resolve_catalog_item_name(raw, fallback=path.stem)
        core_method_name = (
            None if core_method_resolver is None else core_method_resolver(path, raw)
        )
        items.append(
            CatalogItemPayload(
                item_name=item_name,
                display_name=item_name,
                item_kind=item_kind,
                family_name=family_name,
                core_method_name=core_method_name,
                variant_profile_name=item_name,
                preset_group=preset_group,
                source_of_truth=relative_repo_path(path),
                source_kind="hydra_config_group",
                compile_support="preset_selector",
                supported_runtime_paths=supported_runtime_paths,
                declared_fields=declared_fields(raw),
                override_fields=extract_override_fields(raw),
                tags=() if tag_resolver is None else tag_resolver(path, raw),
                metadata=(
                    extract_scalar_metadata(raw, metadata_keys=metadata_keys)
                    if metadata_resolver is None
                    else metadata_resolver(path, raw, metadata_keys)
                ),
            )
        )
    return CatalogSectionPayload(
        section_name=section_name,
        display_name=display_name,
        item_kind=item_kind,
        description=description,
        source_of_truth=relative_dir,
        source_kind="hydra_config_group",
        items=tuple(items),
    )


def build_registry_section(
    *,
    section_name: str,
    display_name: str,
    item_kind: str,
    description: str,
    source_module_name: str,
    entries: Iterable[RegistryCatalogEntry],
    source_of_truth_for_module: SourceOfTruthForModule,
    supported_runtime_paths: tuple[str, ...] | None = None,
    runtime_path_resolver: RuntimePathResolver | None = None,
) -> CatalogSectionPayload:
    """Registry entry 목록을 metadata-only catalog section으로 직렬화한다."""

    if supported_runtime_paths is None and runtime_path_resolver is None:
        raise ValueError(
            "Registry section builder needs supported_runtime_paths or resolver."
        )
    items = tuple(
        build_registry_catalog_item(
            entry=entry,
            item_kind=item_kind,
            supported_runtime_paths=(
                supported_runtime_paths
                if runtime_path_resolver is None
                else runtime_path_resolver(entry)
            ),
            source_of_truth_for_module=source_of_truth_for_module,
        )
        for entry in entries
    )
    return CatalogSectionPayload(
        section_name=section_name,
        display_name=display_name,
        item_kind=item_kind,
        description=description,
        source_of_truth=source_of_truth_for_module(source_module_name),
        source_kind="python_registry",
        items=items,
    )


def build_adapter_family_section(
    *,
    family_metadata: Iterable[SharedAdapterFamilyMetadata],
    source_of_truth_for_module: SourceOfTruthForModule,
    supported_runtime_paths: tuple[str, ...],
) -> CatalogSectionPayload:
    """shared adapter family metadata를 별도 section으로 노출한다."""

    source_of_truth = source_of_truth_for_module(
        "shared.src.config.adapter_family_metadata"
    )
    items = tuple(
        CatalogItemPayload(
            item_name=metadata.family_name,
            display_name=metadata.family_name,
            item_kind="adapter_family",
            family_name=metadata.family_name,
            core_method_name=metadata.family_name,
            variant_profile_name=metadata.family_name,
            source_of_truth=source_of_truth,
            source_kind="python_module",
            compile_support="metadata_only",
            compile_blocker_reason=PHASE2_METADATA_ONLY_BLOCKER,
            supported_adapter_kinds=(metadata.adapter_kind,),
            supported_runtime_paths=supported_runtime_paths,
            accepted_payload_formats=metadata.accepted_update_payload_formats,
            metadata={
                "canonical_update_payload_format": (
                    metadata.canonical_update_payload_format
                ),
            },
        )
        for metadata in family_metadata
    )
    return CatalogSectionPayload(
        section_name="adapter_families",
        display_name="Adapter Families",
        item_kind="adapter_family",
        description="server/agent가 공통으로 해석하는 shared adapter family.",
        source_of_truth=source_of_truth,
        source_kind="python_module",
        items=items,
    )


def build_registry_catalog_item(
    *,
    entry: RegistryCatalogEntry,
    item_kind: str,
    supported_runtime_paths: tuple[str, ...],
    source_of_truth_for_module: SourceOfTruthForModule,
) -> CatalogItemPayload:
    """Registry entry 하나를 metadata-only item으로 직렬화한다."""

    return CatalogItemPayload(
        item_name=entry.item_name,
        display_name=entry.display_name,
        item_kind=item_kind,
        family_name=entry.family_name,
        core_method_name=entry.core_method_name,
        variant_profile_name=entry.item_name,
        source_of_truth=source_of_truth_for_module(entry.implementation_module),
        source_kind="python_registry",
        compile_support="metadata_only",
        compile_blocker_reason=PHASE2_METADATA_ONLY_BLOCKER,
        supported_adapter_kinds=entry.supported_adapter_kinds,
        supported_runtime_paths=supported_runtime_paths,
        accepted_payload_formats=entry.accepted_payload_formats,
        tags=entry.tags,
        metadata=dict(entry.metadata),
    )

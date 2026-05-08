"""Workspace manifest를 Hydra/script preview로 번역하는 compiler service."""

from __future__ import annotations

from dataclasses import dataclass, field

from main_server.src.services.experiment_workspace.catalog.service import (
    ExperimentCatalogService,
)
from main_server.src.services.experiment_workspace.compiler.catalog_lookup import (
    find_section,
    find_track,
    find_variant_item,
)
from main_server.src.services.experiment_workspace.compiler.contracts import (
    ExperimentCompileContext,
)
from main_server.src.services.experiment_workspace.compiler.default_registry import (
    DEFAULT_EXPERIMENT_COMPILE_POLICY_REGISTRY,
)
from main_server.src.services.experiment_workspace.compiler.hydra_overrides import (
    build_hydra_override,
    merge_group_assignments,
    parse_hydra_override_map,
)
from main_server.src.services.experiment_workspace.compiler.registry import (
    ExperimentCompilePolicyRegistry,
)
from main_server.src.services.experiment_workspace.compiler.selection_compiler import (
    compile_workspace_selections,
)
from shared.src.contracts.workspace_manifest_contracts import (
    ResolvedExperimentPlanPayload,
    WorkspaceManifestPayload,
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
        track = find_track(catalog.tracks, manifest.track_name)
        entrypoints_section = find_section(track.sections, "entrypoints")
        entrypoint_item = find_variant_item(
            entrypoints_section.items,
            manifest.entrypoint_name,
        )
        if entrypoint_item.compile_support != "entrypoint":
            raise ValueError(
                f"Workspace entrypoint is not compileable: {manifest.entrypoint_name}."
            )
        if entrypoint_item.script_path is None:
            raise ValueError(
                "Catalog entrypoint is missing script_path: "
                f"{manifest.entrypoint_name}."
            )

        warnings: list[str] = []
        compiled_selections = compile_workspace_selections(
            track=track,
            selections=manifest.selections,
        )
        hydra_overrides = list(compiled_selections.hydra_overrides)

        for key, value in manifest.global_override_patch.items():
            hydra_overrides.append(build_hydra_override(key=key, value=value))

        effective_groups = merge_group_assignments(
            entrypoint_item.default_groups,
            compiled_selections.selection_default_groups,
        )
        hydra_override_map = parse_hydra_override_map(tuple(hydra_overrides))
        compile_context = ExperimentCompileContext(
            manifest=manifest,
            entrypoint_item=entrypoint_item,
            effective_groups=effective_groups,
            hydra_override_map=hydra_override_map,
            catalog_service=self.catalog_service,
        )
        compile_policy = self.compile_policy_registry.resolve(entrypoint_item.item_name)
        warnings.extend(compile_policy.collect_warnings(context=compile_context))
        compile_policy.validate_requirements(context=compile_context)

        script_path = entrypoint_item.script_path
        command_args = (
            "uv",
            "run",
            "python",
            script_path,
            *compiled_selections.selection_default_groups,
            *hydra_overrides,
        )
        return ResolvedExperimentPlanPayload(
            manifest_id=manifest.manifest_id,
            track_name=manifest.track_name,
            entrypoint_name=manifest.entrypoint_name,
            job_config_path=entrypoint_item.source_of_truth,
            script_path=script_path,
            base_default_groups=entrypoint_item.default_groups,
            selection_default_groups=compiled_selections.selection_default_groups,
            hydra_overrides=tuple(hydra_overrides),
            command_args=command_args,
            resolved_selections=compiled_selections.resolved_selections,
            warnings=tuple(warnings),
        )

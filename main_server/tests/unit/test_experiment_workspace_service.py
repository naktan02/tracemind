"""Experiment workspace service tests."""

from __future__ import annotations

from pathlib import Path

from main_server.src.infrastructure.repositories import (
    experiment_workspace_repository,
)
from main_server.src.services.experiment_workspace.catalog_service import (
    ExperimentCatalogService,
)
from main_server.src.services.experiment_workspace.compiler_service import (
    ExperimentCompilerService,
)
from main_server.src.services.experiment_workspace.workspace_service import (
    ExperimentWorkspaceService,
)
from shared.src.contracts.workspace_manifest_contracts import (
    WorkspaceManifestPayload,
    WorkspaceSelectionPayload,
)


def test_experiment_workspace_service_saves_and_reloads_workspace(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    compiler_service = ExperimentCompilerService(
        catalog_service=ExperimentCatalogService(repo_root=repo_root)
    )
    service = ExperimentWorkspaceService(
        compiler_service=compiler_service,
        workspace_repository=experiment_workspace_repository.ExperimentWorkspaceRepository(
            experiments_root=tmp_path / "experiments"
        ),
    )

    saved = service.save_workspace(
        WorkspaceManifestPayload(
            manifest_id="workspace_manifest_seed",
            track_name="seed",
            entrypoint_name="train_softmax_classifier",
            selections=(
                WorkspaceSelectionPayload(
                    slot_name="dataset_presets",
                    section_name="dataset_presets",
                    variant_profile_name="ourafla",
                    family_name="dataset",
                ),
            ),
        )
    )

    assert saved.workspace_id.startswith("workspace_")
    assert saved.manifest.track_name == "seed"
    assert saved.resolved_plan is not None
    assert (
        saved.resolved_plan.script_path
        == "scripts/experiments/train_softmax_classifier.py"
    )

    recent = service.list_workspaces()
    assert recent[0].workspace_id == saved.workspace_id

    loaded = service.get_workspace(saved.workspace_id)
    assert loaded.manifest == saved.manifest
    assert loaded.resolved_plan == saved.resolved_plan

    service.attach_latest_run(
        saved.workspace_id,
        run_id="run_20260423_000000_000001",
        updated_at=loaded.updated_at,
    )
    refreshed = service.get_workspace(saved.workspace_id)
    assert refreshed.latest_run_id == "run_20260423_000000_000001"

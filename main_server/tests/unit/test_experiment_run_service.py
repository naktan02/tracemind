"""Experiment run service tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from main_server.src.infrastructure.repositories import (
    experiment_workspace_repository,
)
from main_server.src.infrastructure.repositories.experiment_run_repository import (
    ExperimentRunRepository,
)
from main_server.src.services.experiment_workspace.catalog_service import (
    ExperimentCatalogService,
)
from main_server.src.services.experiment_workspace.compiler_service import (
    ExperimentCompilerService,
)
from main_server.src.services.experiment_workspace.payloads import (
    LaunchExperimentRunRequestPayload,
)
from main_server.src.services.experiment_workspace.run_service import (
    ExperimentRunService,
    LocalExperimentProcessHandle,
)
from main_server.src.services.experiment_workspace.workspace_service import (
    ExperimentWorkspaceService,
)
from shared.src.contracts.workspace_manifest_contracts import (
    WorkspaceManifestPayload,
    WorkspaceSelectionPayload,
)


@dataclass
class _FakeProcess:
    pid: int = 4242
    exit_code: int | None = None

    def poll(self) -> int | None:
        return self.exit_code


@dataclass
class _LaunchCapture:
    process: _FakeProcess
    command_args: tuple[str, ...] | None = None
    cwd: Path | None = None
    stdout_log_path: Path | None = None
    stderr_log_path: Path | None = None

    def launch(
        self,
        command_args: tuple[str, ...],
        cwd: Path,
        stdout_log_path: Path,
        stderr_log_path: Path,
    ) -> LocalExperimentProcessHandle:
        self.command_args = command_args
        self.cwd = cwd
        self.stdout_log_path = stdout_log_path
        self.stderr_log_path = stderr_log_path
        stdout_log_path.write_text("stdout\n", encoding="utf-8")
        stderr_log_path.write_text("stderr\n", encoding="utf-8")
        return LocalExperimentProcessHandle(process=self.process)


def test_experiment_run_service_launches_local_run_and_refreshes_status(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    compiler_service = ExperimentCompilerService(
        catalog_service=ExperimentCatalogService(repo_root=repo_root)
    )
    experiments_root = tmp_path / "experiments"
    workspace_service = ExperimentWorkspaceService(
        compiler_service=compiler_service,
        workspace_repository=experiment_workspace_repository.ExperimentWorkspaceRepository(
            experiments_root=experiments_root
        ),
    )
    process = _FakeProcess()
    capture = _LaunchCapture(process=process)
    service = ExperimentRunService(
        compiler_service=compiler_service,
        workspace_service=workspace_service,
        run_repository=ExperimentRunRepository(experiments_root=experiments_root),
        repo_root=repo_root,
        process_launcher=capture.launch,
    )

    saved_workspace = workspace_service.save_workspace(_build_seed_manifest())
    launched = service.launch_run(
        LaunchExperimentRunRequestPayload(
            manifest=saved_workspace.manifest,
            workspace_id=saved_workspace.workspace_id,
        )
    )

    assert launched.run_id.startswith("run_")
    assert launched.workspace_id == saved_workspace.workspace_id
    assert launched.status == "running"
    assert capture.command_args is not None
    assert any(
        argument.startswith("hydra.run.dir=") for argument in capture.command_args
    )
    assert "hydra.output_subdir=.hydra" in capture.command_args
    assert capture.cwd == repo_root
    assert capture.stdout_log_path == Path(launched.stdout_log_path)
    assert capture.stderr_log_path == Path(launched.stderr_log_path)
    assert Path(launched.stdout_log_path).exists()
    assert Path(launched.stderr_log_path).exists()
    assert (
        workspace_service.get_workspace(saved_workspace.workspace_id).latest_run_id
        == launched.run_id
    )

    process.exit_code = 0
    refreshed = service.get_run(launched.run_id)
    assert refreshed.status == "succeeded"
    assert refreshed.finished_at is not None
    assert refreshed.exit_code == 0
    assert refreshed.error_message is None


def test_experiment_run_service_rejects_workspace_manifest_mismatch(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    compiler_service = ExperimentCompilerService(
        catalog_service=ExperimentCatalogService(repo_root=repo_root)
    )
    experiments_root = tmp_path / "experiments"
    workspace_service = ExperimentWorkspaceService(
        compiler_service=compiler_service,
        workspace_repository=experiment_workspace_repository.ExperimentWorkspaceRepository(
            experiments_root=experiments_root
        ),
    )
    capture = _LaunchCapture(process=_FakeProcess())
    service = ExperimentRunService(
        compiler_service=compiler_service,
        workspace_service=workspace_service,
        run_repository=ExperimentRunRepository(experiments_root=experiments_root),
        repo_root=repo_root,
        process_launcher=capture.launch,
    )

    saved_workspace = workspace_service.save_workspace(_build_seed_manifest())
    mismatched_manifest = saved_workspace.manifest.model_copy(
        update={"manifest_id": "seed_workspace_draft_v2"}
    )

    with pytest.raises(ValueError, match="launch manifest가 다릅니다"):
        service.launch_run(
            LaunchExperimentRunRequestPayload(
                manifest=mismatched_manifest,
                workspace_id=saved_workspace.workspace_id,
            )
        )


def test_experiment_run_service_rejects_unsupported_track(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    compiler_service = ExperimentCompilerService(
        catalog_service=ExperimentCatalogService(repo_root=repo_root)
    )
    experiments_root = tmp_path / "experiments"
    workspace_service = ExperimentWorkspaceService(
        compiler_service=compiler_service,
        workspace_repository=experiment_workspace_repository.ExperimentWorkspaceRepository(
            experiments_root=experiments_root
        ),
    )
    capture = _LaunchCapture(process=_FakeProcess())
    service = ExperimentRunService(
        compiler_service=compiler_service,
        workspace_service=workspace_service,
        run_repository=ExperimentRunRepository(experiments_root=experiments_root),
        repo_root=repo_root,
        process_launcher=capture.launch,
    )

    with pytest.raises(ValueError, match="seed/central_adaptation track만 지원"):
        service.launch_run(
            LaunchExperimentRunRequestPayload(
                manifest=WorkspaceManifestPayload(
                    manifest_id="federated_workspace",
                    track_name="federated_runtime",
                    entrypoint_name="run_federated_simulation",
                )
            )
        )


def _build_seed_manifest() -> WorkspaceManifestPayload:
    return WorkspaceManifestPayload(
        manifest_id="seed_workspace",
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

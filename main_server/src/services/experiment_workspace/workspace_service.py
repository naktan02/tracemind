"""Developer experiment workspace save/load service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from main_server.src.infrastructure.repositories import (
    experiment_workspace_repository,
)
from main_server.src.services.experiment_workspace.compiler_service import (
    ExperimentCompilerService,
)
from main_server.src.services.experiment_workspace.payloads import (
    SavedWorkspaceDetailPayload,
    SavedWorkspaceSummaryPayload,
)
from shared.src.contracts.workspace_manifest_contracts import (
    WorkspaceManifestPayload,
    dump_resolved_experiment_plan_payload,
    dump_workspace_manifest_payload,
    load_resolved_experiment_plan_payload,
    load_workspace_manifest_payload,
)

WorkspaceRepository = experiment_workspace_repository.ExperimentWorkspaceRepository
StoredWorkspaceRecord = experiment_workspace_repository.StoredExperimentWorkspaceRecord


@dataclass(slots=True)
class ExperimentWorkspaceService:
    """Workspace manifest/preview를 저장하고 다시 여는 service."""

    compiler_service: ExperimentCompilerService
    workspace_repository: WorkspaceRepository = field(
        default_factory=WorkspaceRepository
    )

    def save_workspace(
        self,
        manifest: WorkspaceManifestPayload,
    ) -> SavedWorkspaceDetailPayload:
        """manifest를 compile 후 저장하고 detail payload를 반환한다."""

        resolved_plan = self.compiler_service.compile_manifest(manifest)
        now = datetime.now(tz=timezone.utc)
        workspace_id = build_experiment_workspace_id(created_at=now)
        workspace_dir = self.workspace_repository.workspace_dir(workspace_id)
        manifest_path = workspace_dir / "workspace_manifest.json"
        resolved_plan_path = workspace_dir / "resolved_plan.json"
        dump_workspace_manifest_payload(manifest_path, manifest)
        dump_resolved_experiment_plan_payload(resolved_plan_path, resolved_plan)

        record = StoredWorkspaceRecord(
            workspace_id=workspace_id,
            manifest_id=manifest.manifest_id,
            track_name=manifest.track_name,
            entrypoint_name=manifest.entrypoint_name,
            created_at=now,
            updated_at=now,
            manifest_path=manifest_path,
            resolved_plan_path=resolved_plan_path,
        )
        self.workspace_repository.save(record)
        return _build_workspace_detail_payload(record)

    def list_workspaces(
        self,
        *,
        limit: int = 20,
    ) -> tuple[SavedWorkspaceSummaryPayload, ...]:
        """최근 saved workspace 목록을 반환한다."""

        return tuple(
            _build_workspace_summary_payload(record)
            for record in self.workspace_repository.list_recent(limit=limit)
        )

    def get_workspace(self, workspace_id: str) -> SavedWorkspaceDetailPayload:
        """saved workspace detail을 반환한다."""

        record = self.workspace_repository.get(workspace_id)
        if record is None:
            raise ValueError(f"Unknown workspace: {workspace_id}.")
        return _build_workspace_detail_payload(record)

    def attach_latest_run(
        self,
        workspace_id: str,
        *,
        run_id: str,
        updated_at: datetime,
    ) -> None:
        """workspace에 최신 run reference를 기록한다."""

        self.workspace_repository.set_latest_run(
            workspace_id,
            run_id=run_id,
            updated_at=updated_at,
        )


def build_experiment_workspace_id(*, created_at: datetime) -> str:
    """saved workspace id를 생성한다."""

    timestamp = created_at.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"workspace_{timestamp}"


def _build_workspace_summary_payload(
    record: StoredWorkspaceRecord,
) -> SavedWorkspaceSummaryPayload:
    return SavedWorkspaceSummaryPayload(
        workspace_id=record.workspace_id,
        manifest_id=record.manifest_id,
        track_name=record.track_name,
        entrypoint_name=record.entrypoint_name,
        created_at=record.created_at,
        updated_at=record.updated_at,
        latest_run_id=record.latest_run_id,
    )


def _build_workspace_detail_payload(
    record: StoredWorkspaceRecord,
) -> SavedWorkspaceDetailPayload:
    return SavedWorkspaceDetailPayload(
        workspace_id=record.workspace_id,
        manifest_id=record.manifest_id,
        track_name=record.track_name,
        entrypoint_name=record.entrypoint_name,
        created_at=record.created_at,
        updated_at=record.updated_at,
        latest_run_id=record.latest_run_id,
        manifest=load_workspace_manifest_payload(record.manifest_path),
        resolved_plan=(
            None
            if record.resolved_plan_path is None
            else load_resolved_experiment_plan_payload(record.resolved_plan_path)
        ),
    )

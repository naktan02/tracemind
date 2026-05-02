"""Developer experiment local run service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from main_server.src.infrastructure.repositories.experiment_run_repository import (
    ExperimentRunRepository,
    StoredExperimentRunRecord,
)
from main_server.src.services.experiment_workspace.compiler_service import (
    ExperimentCompilerService,
)
from main_server.src.services.experiment_workspace.payloads import (
    ExperimentRunPayload,
    LaunchExperimentRunRequestPayload,
)
from main_server.src.services.experiment_workspace.run_result_summary import (
    build_experiment_run_result_summary,
    extract_reported_outputs,
)
from main_server.src.services.experiment_workspace.run_runtime_support import (
    LocalExperimentProcessHandle,
    ProcessLauncher,
    build_experiment_run_id,
    build_experiment_run_paths,
    build_finished_run_record,
    build_running_run_record,
    launch_local_process,
)
from main_server.src.services.experiment_workspace.workspace_service import (
    ExperimentWorkspaceService,
)
from shared.src.contracts.workspace_manifest_contracts import (
    dump_resolved_experiment_plan_payload,
    dump_workspace_manifest_payload,
)

SUPPORTED_LOCAL_RUN_TRACKS = ("seed", "central_adaptation")


@dataclass(slots=True)
class ExperimentRunService:
    """local-only experiment run launch와 상태 추적을 담당한다."""

    compiler_service: ExperimentCompilerService
    workspace_service: ExperimentWorkspaceService
    run_repository: ExperimentRunRepository = field(
        default_factory=ExperimentRunRepository
    )
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4])
    process_launcher: ProcessLauncher = field(
        default_factory=lambda: launch_local_process
    )
    _active_processes: dict[str, LocalExperimentProcessHandle] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.run_repository.mark_running_as_interrupted(
            interrupted_at=datetime.now(tz=timezone.utc),
            error_message=(
                "main_server restart 전에 실행 중이던 local experiment run입니다."
            ),
        )

    def launch_run(
        self,
        request: LaunchExperimentRunRequestPayload,
    ) -> ExperimentRunPayload:
        """manifest를 local subprocess로 실행하고 run payload를 반환한다."""

        manifest = request.manifest
        if manifest.track_name not in SUPPORTED_LOCAL_RUN_TRACKS:
            raise ValueError(
                "Phase 4 local run launcher는 seed/central_adaptation track만 "
                f"지원합니다: {manifest.track_name}."
            )
        saved_workspace = None
        if request.workspace_id is not None:
            saved_workspace = self.workspace_service.get_workspace(
                request.workspace_id
            )
            if saved_workspace.manifest != manifest:
                raise ValueError(
                    "workspace_id와 launch manifest가 다릅니다. "
                    "저장된 workspace를 그대로 실행하거나 새 draft를 다시 저장하세요."
                )

        resolved_plan = self.compiler_service.compile_manifest(manifest)
        now = datetime.now(tz=timezone.utc)
        run_id = build_experiment_run_id(created_at=now)
        run_dir = self.run_repository.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_paths = build_experiment_run_paths(run_dir)

        dump_workspace_manifest_payload(run_paths.manifest_path, manifest)
        dump_resolved_experiment_plan_payload(
            run_paths.resolved_plan_path,
            resolved_plan,
        )

        command_args = (
            *resolved_plan.command_args,
            f"hydra.run.dir={run_paths.artifact_root_path}",
            "hydra.output_subdir=.hydra",
        )
        process_handle = self.process_launcher(
            command_args,
            self.repo_root,
            run_paths.stdout_log_path,
            run_paths.stderr_log_path,
        )
        self._active_processes[run_id] = process_handle

        record = build_running_run_record(
            run_id=run_id,
            workspace_id=request.workspace_id,
            manifest_id=manifest.manifest_id,
            track_name=manifest.track_name,
            entrypoint_name=manifest.entrypoint_name,
            created_at=now,
            script_path=resolved_plan.script_path,
            command_args=tuple(command_args),
            paths=run_paths,
            pid=process_handle.pid,
        )
        self.run_repository.save(record)
        if request.workspace_id is not None:
            self.workspace_service.attach_latest_run(
                request.workspace_id,
                run_id=run_id,
                updated_at=now,
            )
        return self._record_to_payload(record)

    def list_runs(self, *, limit: int = 20) -> tuple[ExperimentRunPayload, ...]:
        """최근 run 목록을 반환한다."""

        records = self.run_repository.list_recent(limit=limit)
        return tuple(self._refresh_record(record) for record in records)

    def get_run(self, run_id: str) -> ExperimentRunPayload:
        """단일 run 상태를 반환한다."""

        record = self.run_repository.get(run_id)
        if record is None:
            raise ValueError(f"Unknown experiment run: {run_id}.")
        return self._refresh_record(record)

    def read_log(self, run_id: str, *, stream_name: str) -> Path:
        """stdout/stderr log 파일 경로를 반환한다."""

        record = self.run_repository.get(run_id)
        if record is None:
            raise ValueError(f"Unknown experiment run: {run_id}.")
        if stream_name == "stdout":
            return record.stdout_log_path
        if stream_name == "stderr":
            return record.stderr_log_path
        raise ValueError(f"Unsupported log stream: {stream_name}.")

    def _refresh_record(
        self,
        record: StoredExperimentRunRecord,
    ) -> ExperimentRunPayload:
        if record.status != "running":
            return self._record_to_payload(record)
        active_process = self._active_processes.get(record.run_id)
        if active_process is None:
            return self._record_to_payload(record)

        exit_code = active_process.poll()
        if exit_code is None:
            return self._record_to_payload(record)

        finished_at = datetime.now(tz=timezone.utc)
        updated_record = build_finished_run_record(
            record,
            finished_at=finished_at,
            exit_code=exit_code,
        )
        self.run_repository.save(updated_record)
        active_process.close()
        self._active_processes.pop(record.run_id, None)
        return self._record_to_payload(updated_record)

    def _record_to_payload(
        self,
        record: StoredExperimentRunRecord,
    ) -> ExperimentRunPayload:
        reported_outputs = extract_reported_outputs(record.stdout_log_path)
        result_summary = build_experiment_run_result_summary(
            reported_outputs=reported_outputs,
            repo_root=self.repo_root,
        )
        return ExperimentRunPayload(
            run_id=record.run_id,
            workspace_id=record.workspace_id,
            manifest_id=record.manifest_id,
            track_name=record.track_name,
            entrypoint_name=record.entrypoint_name,
            status=record.status,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            script_path=record.script_path,
            command_args=record.command_args,
            artifact_root_path=str(record.artifact_root_path),
            stdout_log_path=str(record.stdout_log_path),
            stderr_log_path=str(record.stderr_log_path),
            exit_code=record.exit_code,
            error_message=record.error_message,
            reported_outputs=reported_outputs,
            result_summary=result_summary,
        )

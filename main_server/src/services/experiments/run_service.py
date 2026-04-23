"""Developer experiment local run service."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TextIO, cast

from main_server.src.infrastructure.repositories.experiment_run_repository import (
    ExperimentRunRepository,
    StoredExperimentRunRecord,
)
from main_server.src.services.experiments.compiler_service import (
    ExperimentCompilerService,
)
from main_server.src.services.experiments.payloads import (
    ExperimentRunPayload,
    ExperimentRunStatus,
    LaunchExperimentRunRequestPayload,
)
from main_server.src.services.experiments.workspace_service import (
    ExperimentWorkspaceService,
)
from shared.src.contracts.workspace_manifest_contracts import (
    dump_resolved_experiment_plan_payload,
    dump_workspace_manifest_payload,
)

SUPPORTED_LOCAL_RUN_TRACKS = ("seed", "central_adaptation")


class ProcessLike(Protocol):
    """subprocess와 fake process가 공유하는 최소 surface."""

    pid: int

    def poll(self) -> int | None:
        """process 종료 여부를 반환한다."""


@dataclass(slots=True)
class LocalExperimentProcessHandle:
    """실행 중 process와 열려 있는 로그 스트림을 묶는다."""

    process: ProcessLike
    stdout_handle: TextIO | None = None
    stderr_handle: TextIO | None = None

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def poll(self) -> int | None:
        return self.process.poll()

    def close(self) -> None:
        if self.stdout_handle is not None:
            self.stdout_handle.close()
        if self.stderr_handle is not None:
            self.stderr_handle.close()


ProcessLauncher = Callable[
    [tuple[str, ...], Path, Path, Path],
    LocalExperimentProcessHandle,
]


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
        default_factory=lambda: _launch_local_process
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

        manifest_path = run_dir / "workspace_manifest.json"
        resolved_plan_path = run_dir / "resolved_plan.json"
        artifact_root_path = run_dir / "hydra_run"
        stdout_log_path = run_dir / "stdout.log"
        stderr_log_path = run_dir / "stderr.log"
        dump_workspace_manifest_payload(manifest_path, manifest)
        dump_resolved_experiment_plan_payload(resolved_plan_path, resolved_plan)

        command_args = (
            *resolved_plan.command_args,
            f"hydra.run.dir={artifact_root_path}",
            "hydra.output_subdir=.hydra",
        )
        process_handle = self.process_launcher(
            command_args,
            self.repo_root,
            stdout_log_path,
            stderr_log_path,
        )
        self._active_processes[run_id] = process_handle

        record = StoredExperimentRunRecord(
            run_id=run_id,
            workspace_id=request.workspace_id,
            manifest_id=manifest.manifest_id,
            track_name=manifest.track_name,
            entrypoint_name=manifest.entrypoint_name,
        status="running",
            created_at=now,
            started_at=now,
            script_path=resolved_plan.script_path,
            command_args=tuple(command_args),
            manifest_path=manifest_path,
            resolved_plan_path=resolved_plan_path,
            artifact_root_path=artifact_root_path,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            pid=process_handle.pid,
        )
        self.run_repository.save(record)
        if request.workspace_id is not None:
            self.workspace_service.attach_latest_run(
                request.workspace_id,
                run_id=run_id,
                updated_at=now,
            )
        return _record_to_payload(record)

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
            return _record_to_payload(record)
        active_process = self._active_processes.get(record.run_id)
        if active_process is None:
            return _record_to_payload(record)

        exit_code = active_process.poll()
        if exit_code is None:
            return _record_to_payload(record)

        finished_at = datetime.now(tz=timezone.utc)
        updated_record = StoredExperimentRunRecord(
            run_id=record.run_id,
            workspace_id=record.workspace_id,
            manifest_id=record.manifest_id,
            track_name=record.track_name,
            entrypoint_name=record.entrypoint_name,
            status="succeeded" if exit_code == 0 else "failed",
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=finished_at,
            script_path=record.script_path,
            command_args=record.command_args,
            manifest_path=record.manifest_path,
            resolved_plan_path=record.resolved_plan_path,
            artifact_root_path=record.artifact_root_path,
            stdout_log_path=record.stdout_log_path,
            stderr_log_path=record.stderr_log_path,
            pid=record.pid,
            exit_code=exit_code,
            error_message=(
                None if exit_code == 0 else f"Process exited with code {exit_code}."
            ),
        )
        self.run_repository.save(updated_record)
        active_process.close()
        self._active_processes.pop(record.run_id, None)
        return _record_to_payload(updated_record)


def build_experiment_run_id(*, created_at: datetime) -> str:
    """local experiment run id를 생성한다."""

    timestamp = created_at.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"run_{timestamp}"


def _launch_local_process(
    command_args: tuple[str, ...],
    cwd: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
) -> LocalExperimentProcessHandle:
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_log_path.open("w", encoding="utf-8")
    stderr_handle = stderr_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command_args,
        cwd=cwd,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        },
    )
    return LocalExperimentProcessHandle(
        process=process,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
    )


def _record_to_payload(
    record: StoredExperimentRunRecord,
) -> ExperimentRunPayload:
    return ExperimentRunPayload(
        run_id=record.run_id,
        workspace_id=record.workspace_id,
        manifest_id=record.manifest_id,
        track_name=record.track_name,
            entrypoint_name=record.entrypoint_name,
        status=cast(ExperimentRunStatus, record.status),
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
    )

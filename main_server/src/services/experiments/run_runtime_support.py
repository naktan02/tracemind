"""Experiment run runtime support helper."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TextIO, cast

from main_server.src.infrastructure.repositories.experiment_run_repository import (
    StoredExperimentRunRecord,
)
from main_server.src.services.experiments.payloads import (
    ExperimentRunPayload,
    ExperimentRunStatus,
)


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


@dataclass(frozen=True, slots=True)
class ExperimentRunPaths:
    """run 디렉터리 아래 표준 artifact/log 경로."""

    manifest_path: Path
    resolved_plan_path: Path
    artifact_root_path: Path
    stdout_log_path: Path
    stderr_log_path: Path


def build_experiment_run_id(*, created_at: datetime) -> str:
    """local experiment run id를 생성한다."""

    timestamp = created_at.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"run_{timestamp}"


def build_experiment_run_paths(run_dir: Path) -> ExperimentRunPaths:
    """표준 run artifact/log 경로를 조립한다."""

    return ExperimentRunPaths(
        manifest_path=run_dir / "workspace_manifest.json",
        resolved_plan_path=run_dir / "resolved_plan.json",
        artifact_root_path=run_dir / "hydra_run",
        stdout_log_path=run_dir / "stdout.log",
        stderr_log_path=run_dir / "stderr.log",
    )


def launch_local_process(
    command_args: tuple[str, ...],
    cwd: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
) -> LocalExperimentProcessHandle:
    """로컬 실험 subprocess와 로그 핸들을 생성한다."""

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


def build_running_run_record(
    *,
    run_id: str,
    workspace_id: str | None,
    manifest_id: str,
    track_name: str,
    entrypoint_name: str,
    created_at: datetime,
    script_path: str,
    command_args: tuple[str, ...],
    paths: ExperimentRunPaths,
    pid: int,
) -> StoredExperimentRunRecord:
    """launch 직후 저장할 running record를 만든다."""

    return StoredExperimentRunRecord(
        run_id=run_id,
        workspace_id=workspace_id,
        manifest_id=manifest_id,
        track_name=track_name,
        entrypoint_name=entrypoint_name,
        status="running",
        created_at=created_at,
        started_at=created_at,
        script_path=script_path,
        command_args=command_args,
        manifest_path=paths.manifest_path,
        resolved_plan_path=paths.resolved_plan_path,
        artifact_root_path=paths.artifact_root_path,
        stdout_log_path=paths.stdout_log_path,
        stderr_log_path=paths.stderr_log_path,
        pid=pid,
    )


def build_finished_run_record(
    record: StoredExperimentRunRecord,
    *,
    finished_at: datetime,
    exit_code: int,
) -> StoredExperimentRunRecord:
    """종료된 process 상태를 저장소 record로 반영한다."""

    return StoredExperimentRunRecord(
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


def record_to_payload(record: StoredExperimentRunRecord) -> ExperimentRunPayload:
    """저장소 record를 API payload로 변환한다."""

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

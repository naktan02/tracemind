"""Current task runtime result models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TrainingTaskRunStatus(StrEnum):
    """run-current-task 실행 결과 상태."""

    NO_ACTIVE_TASK = "no_active_task"
    INSUFFICIENT_EXAMPLES = "insufficient_examples"
    UPLOADED = "uploaded"
    UNSUPPORTED_RUNTIME = "unsupported_runtime"
    NO_ACTIVE_SHARED_STATE = "no_active_shared_state"
    STALE_SHARED_STATE = "stale_shared_state"
    STALE_RUNTIME_PROFILE = "stale_runtime_profile"


@dataclass(slots=True)
class TrainingTaskRunResult:
    """run-current-task 한 번의 결과 요약."""

    status: TrainingTaskRunStatus | str
    round_id: str | None = None
    task_id: str | None = None
    update_id: str | None = None
    example_count: int = 0
    accepted_count: int = 0
    message: str = ""

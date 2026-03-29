"""학습 task/update/feedback payload와 직렬화 유틸리티."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TrainingTaskPayload(BaseModel):
    """중앙이 로컬에 배포하는 학습 작업 payload."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    task_id: str
    round_id: str
    model_id: str
    model_revision: str
    task_type: str
    training_scope: str
    local_epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    max_steps: int = Field(ge=1)
    objective_config: dict[str, str | int | float | bool]
    selection_policy: dict[str, str | int | float | bool]
    deadline_at: datetime | None = None
    gradient_clip_norm: float | None = Field(default=None, gt=0.0)
    min_required_examples: int | None = Field(default=None, ge=1)
    secure_aggregation_required: bool = False
    notes: str | None = None


class TrainingUpdateEnvelopePayload(BaseModel):
    """로컬 agent가 중앙에 보내는 update envelope payload."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    update_id: str
    round_id: str
    task_id: str
    model_id: str
    base_model_revision: str
    training_scope: str
    payload_ref: str
    payload_format: str
    example_count: int = Field(ge=0)
    client_metrics: dict[str, float]
    created_at: datetime | None = None
    clipped: bool | None = None
    dp_applied: bool | None = None
    checksum: str | None = None
    notes: str | None = None


class DecisionFeedbackSignalPayload(BaseModel):
    """로컬 학습에 사용하는 feedback signal payload."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    signal_id: str
    signal_type: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    occurred_at: datetime
    source_event_ref: str | None = None
    task_context: dict[str, str | int | float | bool] = Field(default_factory=dict)
    notes: str | None = None


def _dump_payload(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def load_training_task_payload(path: Path) -> TrainingTaskPayload:
    """JSON 파일에서 training task payload를 읽는다."""
    return TrainingTaskPayload.model_validate_json(path.read_text(encoding="utf-8"))


def dump_training_task_payload(path: Path, payload: TrainingTaskPayload) -> None:
    """training task payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_training_update_envelope_payload(
    path: Path,
) -> TrainingUpdateEnvelopePayload:
    """JSON 파일에서 update envelope payload를 읽는다."""
    return TrainingUpdateEnvelopePayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_training_update_envelope_payload(
    path: Path,
    payload: TrainingUpdateEnvelopePayload,
) -> None:
    """update envelope payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_decision_feedback_signal_payload(
    path: Path,
) -> DecisionFeedbackSignalPayload:
    """JSON 파일에서 feedback signal payload를 읽는다."""
    return DecisionFeedbackSignalPayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_decision_feedback_signal_payload(
    path: Path,
    payload: DecisionFeedbackSignalPayload,
) -> None:
    """feedback signal payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)

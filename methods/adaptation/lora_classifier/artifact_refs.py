"""LoRA-classifier agent-local artifact ref rules."""

from __future__ import annotations

from datetime import datetime, timezone

from shared.src.contracts.training_contracts import TrainingTask


def build_lora_classifier_base_artifact_ref(
    *,
    prefix: str,
    training_task: TrainingTask,
    created_at: datetime,
) -> str:
    timestamp = utc_timestamp(created_at)
    return "/".join(
        (
            prefix.rstrip("/"),
            slug_artifact_ref_part(training_task.round_id),
            slug_artifact_ref_part(training_task.task_id),
            timestamp,
        )
    )


def utc_timestamp(value: datetime) -> str:
    effective = (
        value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    )
    return effective.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def slug_artifact_ref_part(value: str) -> str:
    normalized = value.strip().replace("/", "_")
    if not normalized:
        raise ValueError("artifact ref path parts must not be empty.")
    return normalized

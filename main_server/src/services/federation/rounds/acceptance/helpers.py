"""Round acceptance helper functions."""

from __future__ import annotations

from datetime import datetime

from main_server.src.services.federation.rounds.acceptance.errors import (
    RoundValidationError,
)
from main_server.src.services.federation.rounds.boundary.models import RoundRecord
from shared.src.contracts.training_contracts import TrainingUpdateEnvelope


def normalize_update(
    update: TrainingUpdateEnvelope,
    *,
    accepted_at: datetime,
) -> TrainingUpdateEnvelope:
    if update.created_at is not None:
        return update
    return update.model_copy(update={"created_at": accepted_at})


def validate_update_context(
    *,
    record: RoundRecord,
    update: TrainingUpdateEnvelope,
) -> None:
    if update.round_id != record.round_id:
        raise RoundValidationError(
            f"Update round_id does not match target round: {update.round_id}"
        )
    if update.task_id != record.training_task.task_id:
        raise RoundValidationError(
            "Update task_id does not match the active training task: "
            f"{update.task_id}"
        )
    if update.model_id != record.active_manifest.model_id:
        raise RoundValidationError(
            "Update model_id does not match the active manifest: "
            f"{update.model_id}"
        )
    if update.base_model_revision != record.active_manifest.model_revision:
        raise RoundValidationError(
            "Update base_model_revision does not match the active manifest: "
            f"{update.base_model_revision}"
        )
    if update.training_scope != record.training_task.training_scope:
        raise RoundValidationError(
            "Update training_scope does not match the active training task: "
            f"{update.training_scope}"
        )


def find_existing_update(
    *,
    record: RoundRecord,
    update_id: str,
) -> TrainingUpdateEnvelope | None:
    for existing in record.updates:
        if existing.update_id == update_id:
            return existing
    return None


def find_existing_agent_update(
    *,
    record: RoundRecord,
    agent_id: str,
) -> TrainingUpdateEnvelope | None:
    for existing in record.updates:
        if existing.agent_id == agent_id:
            return existing
    return None


def build_idempotency_fingerprint(
    update: TrainingUpdateEnvelope,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    int,
    tuple[tuple[str, float], ...],
    bool | None,
    bool | None,
    str | None,
    str | None,
]:
    return (
        update.schema_version,
        update.update_id,
        update.round_id,
        update.task_id,
        update.model_id,
        update.base_model_revision,
        update.training_scope,
        update.example_count,
        tuple(sorted(update.client_metrics.items())),
        update.clipped,
        update.dp_applied,
        update.checksum,
        update.payload_ref,
    )

"""PEFT text encoder report artifact compatibility helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from methods.adaptation.update_family_report_artifacts import (
    PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME,
    aggregate_snapshot_candidates,
    objective_value,
    primary_update_ref_fields,
)

_PEFT_ENCODER_UPDATE_FAMILY_NAME = PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME


def peft_encoder_objective_value(
    objective: Mapping[str, object],
    key: str,
) -> object:
    """기존 호출부를 위한 PEFT text encoder objective alias."""

    return objective_value(
        update_family_name=_PEFT_ENCODER_UPDATE_FAMILY_NAME,
        objective=objective,
        key=key,
    )


def peft_encoder_primary_update_ref_fields(
    update_payload: Mapping[str, object],
) -> tuple[str, ...]:
    """기존 호출부를 위한 PEFT text encoder update ref alias."""

    return primary_update_ref_fields(
        update_family_name=_PEFT_ENCODER_UPDATE_FAMILY_NAME,
        update_payload=update_payload,
    )


def peft_encoder_aggregate_snapshot_candidates(
    *,
    artifact_root: Path,
    model_revision: str,
) -> tuple[tuple[Path, ...], ...]:
    """기존 호출부를 위한 PEFT text encoder snapshot alias."""

    return aggregate_snapshot_candidates(
        update_family_name=_PEFT_ENCODER_UPDATE_FAMILY_NAME,
        artifact_root=artifact_root,
        model_revision=model_revision,
    )

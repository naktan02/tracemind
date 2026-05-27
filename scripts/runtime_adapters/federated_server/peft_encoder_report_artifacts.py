"""PEFT encoder report artifact rules for FL simulation verification."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from methods.adaptation.peft_text_classifier.report_artifacts import (
    classifier_aggregate_snapshot_candidates,
    classifier_objective_value,
    classifier_primary_update_ref_fields,
)


def peft_encoder_objective_value(
    objective: Mapping[str, object],
    key: str,
) -> object:
    """v2 PEFT objective key를 먼저 읽고 legacy LoRA key로 fallback한다."""

    return classifier_objective_value(objective, key)


def peft_encoder_primary_update_ref_fields(
    update_payload: Mapping[str, object],
) -> tuple[str, str]:
    """merged update payload의 adapter/head artifact ref field 쌍을 고른다."""

    return classifier_primary_update_ref_fields(update_payload)


def peft_encoder_aggregate_snapshot_candidates(
    *,
    artifact_root: Path,
    model_revision: str,
) -> tuple[tuple[Path, Path], ...]:
    """final global PEFT-backed classifier snapshot 후보 경로를 반환한다."""

    return classifier_aggregate_snapshot_candidates(
        artifact_root=artifact_root,
        model_revision=model_revision,
    )

"""PEFT text encoder report artifact rules."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
)

PEFT_ENCODER_OBJECTIVE_NAMES = (PEFT_CLASSIFIER_ADAPTER_KIND,)
PEFT_ENCODER_PRIMARY_REF_FIELDS = (
    "peft_adapter_delta_artifact_ref",
    "classifier_head_delta_artifact_ref",
)
PEFT_ENCODER_SNAPSHOT_SPECS = ((PEFT_CLASSIFIER_ADAPTER_KIND, "peft_adapter.json"),)


def peft_encoder_objective_value(
    objective: Mapping[str, object],
    key: str,
) -> object:
    """PEFT encoder objective namespace에서 값을 읽는다."""

    for objective_name in PEFT_ENCODER_OBJECTIVE_NAMES:
        value = _nested_or_flat_value(objective, objective_name, key)
        if value is not None:
            return value
    return None


def peft_encoder_primary_update_ref_fields(
    update_payload: Mapping[str, object],
) -> tuple[str, str]:
    """merged update payload의 adapter/head artifact ref field 쌍을 고른다."""

    return PEFT_ENCODER_PRIMARY_REF_FIELDS


def peft_encoder_aggregate_snapshot_candidates(
    *,
    artifact_root: Path,
    model_revision: str,
) -> tuple[tuple[Path, Path], ...]:
    """final global PEFT encoder snapshot 후보 경로를 반환한다."""

    return tuple(
        (
            artifact_root / family_name / model_revision / adapter_artifact_name,
            artifact_root / family_name / model_revision / "classifier_head.json",
        )
        for family_name, adapter_artifact_name in PEFT_ENCODER_SNAPSHOT_SPECS
    )


def _nested_or_flat_value(
    payload: Mapping[str, object],
    namespace: str,
    key: str,
) -> object:
    flat_value = payload.get(f"{namespace}.{key}")
    if flat_value is not None:
        return flat_value
    nested = payload.get(namespace)
    return nested.get(key) if isinstance(nested, Mapping) else None

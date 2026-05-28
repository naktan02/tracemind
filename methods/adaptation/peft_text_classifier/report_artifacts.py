"""PEFT-backed classifier report artifact compatibility rules."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
)

LEGACY_LORA_CLASSIFIER_ADAPTER_KIND = "lora_classifier"
CLASSIFIER_OBJECTIVE_NAMES = (
    PEFT_CLASSIFIER_ADAPTER_KIND,
    LEGACY_LORA_CLASSIFIER_ADAPTER_KIND,
)
PEFT_CLASSIFIER_PRIMARY_REF_FIELDS = (
    "peft_adapter_delta_artifact_ref",
    "classifier_head_delta_artifact_ref",
)
LEGACY_LORA_CLASSIFIER_PRIMARY_REF_FIELDS = (
    "peft_adapter_delta_artifact_ref",
    "classifier_head_delta_artifact_ref",
)
CLASSIFIER_SNAPSHOT_SPECS = (
    (PEFT_CLASSIFIER_ADAPTER_KIND, "peft_adapter.json"),
    (LEGACY_LORA_CLASSIFIER_ADAPTER_KIND, "lora_adapter.json"),
)


def classifier_objective_value(
    objective: Mapping[str, object],
    key: str,
) -> object:
    """v2 PEFT objective key를 먼저 읽고 legacy LoRA key로 fallback한다."""

    for objective_name in CLASSIFIER_OBJECTIVE_NAMES:
        value = _nested_or_flat_value(objective, objective_name, key)
        if value is not None:
            return value
    return None


def classifier_primary_update_ref_fields(
    update_payload: Mapping[str, object],
) -> tuple[str, str]:
    """merged update payload의 adapter/head artifact ref field 쌍을 고른다."""

    if update_payload.get("peft_adapter_delta_artifact_ref") is not None:
        return PEFT_CLASSIFIER_PRIMARY_REF_FIELDS
    return LEGACY_LORA_CLASSIFIER_PRIMARY_REF_FIELDS


def classifier_aggregate_snapshot_candidates(
    *,
    artifact_root: Path,
    model_revision: str,
) -> tuple[tuple[Path, Path], ...]:
    """final global classifier snapshot 후보 경로를 v2/v1 순서로 반환한다."""

    return tuple(
        (
            artifact_root / family_name / model_revision / adapter_artifact_name,
            artifact_root / family_name / model_revision / "classifier_head.json",
        )
        for family_name, adapter_artifact_name in CLASSIFIER_SNAPSHOT_SPECS
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

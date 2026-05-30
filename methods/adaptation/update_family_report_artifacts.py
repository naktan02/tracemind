"""Update family별 FL report artifact 검증 규칙."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
)


@dataclass(frozen=True, slots=True)
class AggregateSnapshotSpec:
    """final aggregate snapshot을 구성하는 artifact 경로 규칙."""

    artifact_family_name: str
    artifact_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UpdateFamilyReportArtifactRules:
    """report verifier가 update family별로 알아야 하는 artifact 표면."""

    objective_names: tuple[str, ...]
    primary_update_ref_fields: tuple[str, ...]
    aggregate_snapshot_specs: tuple[AggregateSnapshotSpec, ...] = ()


PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME = "peft_text_encoder"

_PEFT_TEXT_ENCODER_REPORT_RULES = UpdateFamilyReportArtifactRules(
    objective_names=(PEFT_CLASSIFIER_ADAPTER_KIND,),
    primary_update_ref_fields=(
        "peft_adapter_delta_artifact_ref",
        "classifier_head_delta_artifact_ref",
    ),
    aggregate_snapshot_specs=(
        AggregateSnapshotSpec(
            artifact_family_name=PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME,
            artifact_names=("peft_adapter.json", "classifier_head.json"),
        ),
    ),
)

_RULES_BY_UPDATE_FAMILY = {
    PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME: _PEFT_TEXT_ENCODER_REPORT_RULES,
}


def objective_value(
    *,
    update_family_name: str | None,
    objective: Mapping[str, object],
    key: str,
) -> object:
    """update family objective namespace에서 값을 읽는다."""

    for rules in _candidate_rules(update_family_name):
        for objective_name in rules.objective_names:
            value = _nested_or_flat_value(objective, objective_name, key)
            if value is not None:
                return value
    return None


def primary_update_ref_fields(
    *,
    update_family_name: str | None,
    update_payload: Mapping[str, object],
) -> tuple[str, ...]:
    """merged update payload에서 server-owned ref로 검증할 field를 고른다."""

    rules = _rules_for_update_family(update_family_name)
    return rules.primary_update_ref_fields


def aggregate_snapshot_candidates(
    *,
    update_family_name: str,
    artifact_root: Path,
    model_revision: str,
) -> tuple[tuple[Path, ...], ...]:
    """final global aggregate snapshot 후보 경로를 반환한다."""

    rules = _rules_for_update_family(update_family_name)
    return tuple(
        tuple(
            artifact_root
            / spec.artifact_family_name
            / model_revision
            / artifact_name
            for artifact_name in spec.artifact_names
        )
        for spec in rules.aggregate_snapshot_specs
    )


def _candidate_rules(
    update_family_name: str | None,
) -> Iterable[UpdateFamilyReportArtifactRules]:
    if update_family_name is None:
        return tuple(_RULES_BY_UPDATE_FAMILY.values())
    return (_rules_for_update_family(update_family_name),)


def _rules_for_update_family(
    update_family_name: str | None,
) -> UpdateFamilyReportArtifactRules:
    if update_family_name is None:
        if len(_RULES_BY_UPDATE_FAMILY) == 1:
            return next(iter(_RULES_BY_UPDATE_FAMILY.values()))
        raise ValueError("update_family_name is required for artifact field rules.")
    try:
        return _RULES_BY_UPDATE_FAMILY[update_family_name]
    except KeyError as error:
        raise ValueError(
            f"No FL report artifact rules for update family {update_family_name!r}."
        ) from error


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

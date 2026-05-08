"""Hydra selector/override string utilities for experiment compiler."""

from __future__ import annotations

from shared.src.contracts.workspace_manifest_contracts import WorkspaceConfigScalar


def build_hydra_override(
    *,
    key: str,
    value: WorkspaceConfigScalar,
) -> str:
    return f"{key}={format_hydra_value(value)}"


def build_preset_hydra_override(
    *,
    preset_group: str,
    key: str,
    value: WorkspaceConfigScalar,
) -> str:
    return build_hydra_override(
        key=f"{preset_group}.{key}",
        value=value,
    )


def format_hydra_value(value: WorkspaceConfigScalar) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def merge_group_assignments(*group_sets: tuple[str, ...]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for group_set in group_sets:
        for entry in group_set:
            if "=" not in entry:
                continue
            key, value = entry.split("=", 1)
            merged[key] = value
    return merged


def parse_hydra_override_map(overrides: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        parsed[key] = value
    return parsed

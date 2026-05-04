"""Federated aggregation method metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FederatedAggregationMethodSpec:
    """server lifecycle과 분리된 aggregation method catalog 항목."""

    adapter_kind: str
    method_name: str
    implementation_module: str
    core_function_name: str
    aliases: tuple[str, ...] = ()

"""FL SSL method-local client diagnostics discovery tests."""

from __future__ import annotations

from methods.federated_ssl.client_diagnostics import (
    client_method_diagnostics_summary_payload,
    known_client_diagnostic_metric_names,
)
from methods.federated_ssl.fedmatch.client_diagnostics import FEDMATCH_HELPER_COUNT


def test_client_diagnostics_discovers_method_local_metric_contracts() -> None:
    names = known_client_diagnostic_metric_names()

    assert FEDMATCH_HELPER_COUNT in names


def test_client_diagnostics_summary_uses_discovered_method_builder() -> None:
    payload = client_method_diagnostics_summary_payload(
        ({FEDMATCH_HELPER_COUNT: 2.0}, {FEDMATCH_HELPER_COUNT: 4.0}),
        numeric_summary=lambda values: {
            "count": len(tuple(values)),
            "mean": sum(values) / len(tuple(values)) if values else None,
        },
    )

    assert payload[f"{FEDMATCH_HELPER_COUNT}_summary"] == {
        "count": 2,
        "mean": 3.0,
    }

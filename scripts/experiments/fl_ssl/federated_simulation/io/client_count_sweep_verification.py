"""FL SSL client-count sweep summary verification."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

from .report_verification_helpers import (
    expect_equal,
    load_json_object,
    object_mapping,
    object_sequence,
    optional_int,
    resolve_report_path,
)
from .report_verification_models import (
    FederatedReportExpectation,
    VerificationResult,
)


def verify_client_count_sweep_summary_path(
    summary_path: Path,
    *,
    expected_client_counts: tuple[int, ...],
    report_expectation: FederatedReportExpectation,
) -> VerificationResult:
    summary = load_json_object(summary_path)
    errors = list(
        _verify_client_count_sweep_summary_payload(
            summary=summary,
            expected_client_counts=expected_client_counts,
            report_expectation=report_expectation,
        )
    )
    errors.extend(
        _verify_client_count_sweep_run_reports(
            summary=summary,
            summary_path=summary_path,
            report_expectation=report_expectation,
        )
    )
    return VerificationResult(artifact=str(summary_path), errors=tuple(errors))


def _verify_client_count_sweep_summary_payload(
    *,
    summary: Mapping[str, object],
    expected_client_counts: tuple[int, ...],
    report_expectation: FederatedReportExpectation,
) -> tuple[str, ...]:
    errors: list[str] = []
    expect_equal(
        errors,
        "schema_version",
        summary.get("schema_version"),
        "fl_ssl_client_count_sweep_summary.v1",
    )
    observed_counts = tuple(
        int(value) for value in object_sequence(summary.get("client_counts"))
    )
    if observed_counts != expected_client_counts:
        errors.append(
            "client_counts expected "
            f"{list(expected_client_counts)!r}, got {list(observed_counts)!r}."
        )
    expect_equal(
        errors,
        "protocol.round_budget",
        object_mapping(summary.get("protocol")).get("round_budget"),
        report_expectation.expected_round_budget,
    )
    runs = object_sequence(summary.get("runs"))
    if len(runs) != len(expected_client_counts):
        errors.append(
            f"runs length expected {len(expected_client_counts)}, got {len(runs)}."
        )
    return tuple(errors)


def _verify_client_count_sweep_run_reports(
    *,
    summary: Mapping[str, object],
    summary_path: Path,
    report_expectation: FederatedReportExpectation,
) -> tuple[str, ...]:
    from .report_verification import (
        verify_federated_simulation_report_path,
    )

    errors: list[str] = []
    for run in object_sequence(summary.get("runs")):
        run_payload = object_mapping(run)
        client_count = optional_int(run_payload.get("client_count"))
        report_path = resolve_report_path(summary_path, run_payload.get("report_path"))
        if report_path is None:
            errors.append(f"client_count={client_count}: report_path is missing.")
            continue
        if not report_path.exists():
            errors.append(
                f"client_count={client_count}: report does not exist: {report_path}."
            )
            continue
        result = verify_federated_simulation_report_path(
            report_path,
            replace(report_expectation, expected_client_count=client_count),
        )
        errors.extend(
            f"client_count={client_count}: {error}" for error in result.errors
        )
    return tuple(errors)

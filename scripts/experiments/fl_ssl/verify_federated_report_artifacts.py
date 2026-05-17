"""이미 생성된 FL SSL report artifact를 검증하는 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from scripts.experiments.fl_ssl.federated_simulation.io.report_verification import (
    FederatedReportExpectation,
    VerificationResult,
    verify_client_count_sweep_summary_path,
    verify_federated_simulation_report_path,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    expectation = FederatedReportExpectation(
        expected_completed_rounds=args.expected_completed_rounds,
        expected_round_budget=args.expected_round_budget,
        expected_client_count=args.expected_client_count,
        expected_ssl_algorithm=args.expected_ssl_algorithm,
        expected_ssl_method=args.expected_ssl_method,
        expected_adapter_family=args.expected_adapter_family,
        expected_aggregation=args.expected_aggregation,
        expected_delta_format=args.expected_delta_format,
        expected_round_record_count=args.expected_round_record_count,
        expected_round_update_count=args.expected_round_update_count,
        expected_round_update_count_matches_client_count=(
            args.expect_round_update_count_matches_client_count
        ),
        expected_embedding_metadata_status=args.expected_embedding_metadata_status,
        expected_embedding_backend=args.expected_embedding_backend,
        expected_embedding_model_id=args.expected_embedding_model_id,
        expected_embedding_device=args.expected_embedding_device,
        expected_embedding_local_files_only=_parse_optional_bool(
            args.expected_embedding_local_files_only
        ),
        expected_local_trainer_metadata_status=(
            args.expected_local_trainer_metadata_status
        ),
        expected_local_trainer_device=args.expected_local_trainer_device,
        expected_local_trainer_local_files_only=_parse_optional_bool(
            args.expected_local_trainer_local_files_only
        ),
    )

    results: list[VerificationResult] = []
    for report_path in args.report:
        results.append(
            verify_federated_simulation_report_path(
                Path(report_path),
                expectation,
            )
        )
    for summary_path in args.client_count_sweep_summary:
        if args.expected_client_counts is None:
            parser.error(
                "--client-count-sweep-summary requires --expected-client-counts."
            )
        results.append(
            verify_client_count_sweep_summary_path(
                Path(summary_path),
                expected_client_counts=_parse_int_csv(args.expected_client_counts),
                report_expectation=expectation,
            )
        )

    if not results:
        parser.error(
            "At least one --report or --client-count-sweep-summary is required."
        )
    for result in results:
        _print_result(result)
    return 0 if all(result.passed for result in results) else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify existing FL SSL report artifacts without running training."
        ),
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="fl_ssl_main_comparison.report.json path. Can be repeated.",
    )
    parser.add_argument(
        "--client-count-sweep-summary",
        action="append",
        default=[],
        help="fl_ssl_client_count_sweep.summary.json path. Can be repeated.",
    )
    parser.add_argument(
        "--expected-client-counts",
        help="Comma-separated expected client_count values for sweep summary.",
    )
    parser.add_argument("--expected-completed-rounds", type=int)
    parser.add_argument("--expected-round-budget", type=int)
    parser.add_argument("--expected-client-count", type=int)
    parser.add_argument("--expected-ssl-algorithm")
    parser.add_argument("--expected-ssl-method")
    parser.add_argument("--expected-adapter-family")
    parser.add_argument("--expected-aggregation")
    parser.add_argument("--expected-delta-format")
    parser.add_argument("--expected-round-record-count", type=int)
    parser.add_argument("--expected-round-update-count", type=int)
    parser.add_argument(
        "--expect-round-update-count-matches-client-count",
        action="store_true",
    )
    parser.add_argument("--expected-embedding-metadata-status")
    parser.add_argument("--expected-embedding-backend")
    parser.add_argument("--expected-embedding-model-id")
    parser.add_argument("--expected-embedding-device")
    parser.add_argument(
        "--expected-embedding-local-files-only",
        choices=("true", "false"),
    )
    parser.add_argument("--expected-local-trainer-metadata-status")
    parser.add_argument("--expected-local-trainer-device")
    parser.add_argument(
        "--expected-local-trainer-local-files-only",
        choices=("true", "false"),
    )
    return parser


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "true"


def _print_result(result: VerificationResult) -> None:
    if result.passed:
        print(f"PASS {result.artifact}")
        return
    print(f"FAIL {result.artifact}")
    for error in result.errors:
        print(f"  - {error}")


if __name__ == "__main__":
    raise SystemExit(main())

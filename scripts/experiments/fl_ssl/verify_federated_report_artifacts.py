"""이미 생성된 FL SSL report artifact를 검증하는 CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Mapping, Sequence

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
        expected_seed=args.expected_seed,
        expected_shard_policy_name=args.expected_shard_policy_name,
        expected_shard_alpha=args.expected_shard_alpha,
        expected_split_id=args.expected_split_id,
        expected_split_id_contains=args.expected_split_id_contains,
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
    for manifest_path in args.manifest:
        results.extend(_verify_manifest_path(Path(manifest_path)))
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
            "At least one --manifest, --report, or "
            "--client-count-sweep-summary is required."
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
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help=(
            "JSON manifest containing artifact paths and per-artifact expectations. "
            "Can be repeated."
        ),
    )
    parser.add_argument("--expected-completed-rounds", type=int)
    parser.add_argument("--expected-round-budget", type=int)
    parser.add_argument("--expected-client-count", type=int)
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-shard-policy-name")
    parser.add_argument("--expected-shard-alpha", type=float)
    parser.add_argument("--expected-split-id")
    parser.add_argument("--expected-split-id-contains")
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


def _verify_manifest_path(manifest_path: Path) -> list[VerificationResult]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _manifest_entries(manifest)
    return [
        _verify_manifest_entry(
            manifest_path=manifest_path,
            entry=entry,
            index=index,
        )
        for index, entry in enumerate(entries, start=1)
    ]


def _manifest_entries(manifest: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(manifest, list):
        raw_entries = manifest
    elif isinstance(manifest, Mapping):
        raw_entries = manifest.get("artifacts")
    else:
        raw_entries = None
    if not isinstance(raw_entries, list):
        raise ValueError("Verifier manifest must be a list or contain artifacts list.")
    entries: list[Mapping[str, object]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("Verifier manifest entries must be JSON objects.")
        entries.append(raw_entry)
    return tuple(entries)


def _verify_manifest_entry(
    *,
    manifest_path: Path,
    entry: Mapping[str, object],
    index: int,
) -> VerificationResult:
    name = str(entry.get("name") or f"artifact_{index}")
    expectation = _expectation_from_manifest_entry(entry)
    report_path = _optional_manifest_path(manifest_path, entry.get("report"))
    summary_path = _optional_manifest_path(
        manifest_path,
        entry.get("client_count_sweep_summary"),
    )
    if report_path is not None and summary_path is not None:
        raise ValueError(
            f"Verifier manifest entry {name!r} must not define both report "
            "and client_count_sweep_summary."
        )
    if report_path is not None:
        result = verify_federated_simulation_report_path(report_path, expectation)
        return _label_result(name=name, result=result)
    if summary_path is not None:
        expected_client_counts = _int_sequence(
            entry.get("expected_client_counts"),
            field_name="expected_client_counts",
        )
        result = verify_client_count_sweep_summary_path(
            summary_path,
            expected_client_counts=expected_client_counts,
            report_expectation=expectation,
        )
        return _label_result(name=name, result=result)
    raise ValueError(
        f"Verifier manifest entry {name!r} requires report or "
        "client_count_sweep_summary."
    )


def _expectation_from_manifest_entry(
    entry: Mapping[str, object],
) -> FederatedReportExpectation:
    raw_expectation = entry.get("expectation", {})
    if not isinstance(raw_expectation, Mapping):
        raise ValueError("Verifier manifest expectation must be a JSON object.")
    allowed_fields = {field.name for field in fields(FederatedReportExpectation)}
    unknown_fields = sorted(set(raw_expectation) - allowed_fields)
    if unknown_fields:
        raise ValueError(f"Unknown verifier expectation fields: {unknown_fields}.")
    return FederatedReportExpectation(**dict(raw_expectation))


def _optional_manifest_path(manifest_path: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return manifest_path.parent / path


def _int_sequence(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Verifier manifest {field_name} must be a non-empty list.")
    return tuple(int(item) for item in value)


def _label_result(*, name: str, result: VerificationResult) -> VerificationResult:
    return VerificationResult(
        artifact=f"{name}: {result.artifact}",
        errors=result.errors,
    )


def _print_result(result: VerificationResult) -> None:
    if result.passed:
        print(f"PASS {result.artifact}")
        return
    print(f"FAIL {result.artifact}")
    for error in result.errors:
        print(f"  - {error}")


if __name__ == "__main__":
    raise SystemExit(main())

"""FL SSL report artifact verification helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class FederatedReportExpectation:
    """이미 생성된 FL SSL report가 맞춰야 하는 실행 metadata 기대값."""

    expected_completed_rounds: int | None = None
    expected_round_budget: int | None = None
    expected_client_count: int | None = None
    expected_seed: int | None = None
    expected_shard_policy_name: str | None = None
    expected_shard_alpha: float | None = None
    expected_split_id: str | None = None
    expected_split_id_contains: str | None = None
    expected_ssl_algorithm: str | None = None
    expected_ssl_method: str | None = None
    expected_adapter_family: str | None = None
    expected_aggregation: str | None = None
    expected_delta_format: str | None = None
    expected_round_record_count: int | None = None
    expected_round_update_count: int | None = None
    expected_round_update_count_matches_client_count: bool = False
    expected_embedding_metadata_status: str | None = None
    expected_embedding_backend: str | None = None
    expected_embedding_model_id: str | None = None
    expected_embedding_device: str | None = None
    expected_embedding_local_files_only: bool | None = None
    expected_local_trainer_metadata_status: str | None = None
    expected_local_trainer_device: str | None = None
    expected_local_trainer_local_files_only: bool | None = None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """검증 대상 artifact와 발견된 오류 목록."""

    artifact: str
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


def verify_federated_simulation_report_path(
    report_path: Path,
    expectation: FederatedReportExpectation,
) -> VerificationResult:
    return verify_federated_simulation_report_payload(
        artifact=str(report_path),
        payload=_load_json_object(report_path),
        expectation=expectation,
    )


def verify_federated_simulation_report_payload(
    *,
    artifact: str,
    payload: Mapping[str, object],
    expectation: FederatedReportExpectation,
) -> VerificationResult:
    errors: list[str] = []
    protocol = _object_mapping(payload.get("protocol"))
    objective = _object_mapping(protocol.get("objective") or payload.get("objective"))
    round_runtime = _object_mapping(
        protocol.get("round_runtime") or payload.get("round_runtime")
    )
    embedding_adapter = _object_mapping(
        protocol.get("embedding_adapter") or payload.get("embedding_adapter")
    )
    shard_policy = _object_mapping(protocol.get("shard_policy"))
    fl_data_source = _object_mapping(protocol.get("fl_data_source"))
    local_trainer_runtime = _object_mapping(
        protocol.get("local_trainer_runtime") or payload.get("local_trainer_runtime")
    )
    rounds = _object_sequence(payload.get("rounds"))

    _expect_equal(
        errors,
        "protocol.completed_rounds",
        protocol.get("completed_rounds"),
        expectation.expected_completed_rounds,
    )
    _expect_equal(
        errors,
        "protocol.round_budget",
        protocol.get("round_budget"),
        expectation.expected_round_budget,
    )
    _expect_equal(
        errors,
        "protocol.client_count",
        protocol.get("client_count"),
        expectation.expected_client_count,
    )
    _expect_equal(
        errors,
        "protocol.seed",
        protocol.get("seed"),
        expectation.expected_seed,
    )
    _expect_equal(
        errors,
        "protocol.shard_policy.name",
        shard_policy.get("name"),
        expectation.expected_shard_policy_name,
    )
    _expect_float_equal(
        errors,
        "protocol.shard_policy.alpha",
        shard_policy.get("alpha"),
        expectation.expected_shard_alpha,
    )
    _expect_equal(
        errors,
        "protocol.fl_data_source.split_id",
        fl_data_source.get("split_id"),
        expectation.expected_split_id,
    )
    _expect_contains(
        errors,
        "protocol.fl_data_source.split_id",
        fl_data_source.get("split_id"),
        expectation.expected_split_id_contains,
    )
    _expect_equal(
        errors,
        "objective.query_ssl.algorithm_name",
        _nested_or_flat_value(objective, "query_ssl", "algorithm_name"),
        expectation.expected_ssl_algorithm,
    )
    _expect_equal(
        errors,
        "objective.query_ssl.method_name",
        _nested_or_flat_value(objective, "query_ssl", "method_name"),
        expectation.expected_ssl_method,
    )
    _expect_equal(
        errors,
        "round_runtime.adapter_family_name",
        round_runtime.get("adapter_family_name"),
        expectation.expected_adapter_family,
    )
    _expect_equal(
        errors,
        "round_runtime.aggregation_backend_name",
        round_runtime.get("aggregation_backend_name"),
        expectation.expected_aggregation,
    )
    _expect_equal(
        errors,
        "objective.lora_classifier.delta_format",
        _nested_or_flat_value(objective, "lora_classifier", "delta_format"),
        expectation.expected_delta_format,
    )
    _expect_equal(
        errors,
        "embedding_adapter.metadata_status",
        embedding_adapter.get("metadata_status"),
        expectation.expected_embedding_metadata_status,
    )
    _expect_equal(
        errors,
        "embedding_adapter.backend",
        embedding_adapter.get("backend"),
        expectation.expected_embedding_backend,
    )
    _expect_equal(
        errors,
        "embedding_adapter.model_id",
        embedding_adapter.get("model_id"),
        expectation.expected_embedding_model_id,
    )
    _expect_equal(
        errors,
        "embedding_adapter.device",
        embedding_adapter.get("device"),
        expectation.expected_embedding_device,
    )
    _expect_equal(
        errors,
        "embedding_adapter.local_files_only",
        embedding_adapter.get("local_files_only"),
        expectation.expected_embedding_local_files_only,
    )
    _expect_equal(
        errors,
        "local_trainer_runtime.metadata_status",
        local_trainer_runtime.get("metadata_status"),
        expectation.expected_local_trainer_metadata_status,
    )
    _expect_equal(
        errors,
        "local_trainer_runtime.device",
        local_trainer_runtime.get("device"),
        expectation.expected_local_trainer_device,
    )
    _expect_equal(
        errors,
        "local_trainer_runtime.local_files_only",
        local_trainer_runtime.get("local_files_only"),
        expectation.expected_local_trainer_local_files_only,
    )
    _verify_round_records(
        errors=errors,
        rounds=rounds,
        protocol=protocol,
        expectation=expectation,
    )
    return VerificationResult(artifact=artifact, errors=tuple(errors))


def _verify_round_records(
    *,
    errors: list[str],
    rounds: tuple[object, ...],
    protocol: Mapping[str, object],
    expectation: FederatedReportExpectation,
) -> None:
    _expect_equal(
        errors,
        "rounds.length",
        len(rounds),
        expectation.expected_round_record_count,
    )
    if (
        expectation.expected_round_update_count is None
        and not expectation.expected_round_update_count_matches_client_count
    ):
        return
    if not rounds:
        errors.append("rounds must not be empty when round update counts are expected.")
        return

    expected_update_count = expectation.expected_round_update_count
    if expectation.expected_round_update_count_matches_client_count:
        expected_client_count = _optional_int(protocol.get("client_count"))
        if expected_client_count is None:
            errors.append(
                "protocol.client_count is required when round update counts "
                "must match client_count."
            )
            return
        expected_update_count = expected_client_count

    for index, round_payload in enumerate(rounds, start=1):
        round_mapping = _object_mapping(round_payload)
        round_label = round_mapping.get("round_id") or f"round_index={index}"
        _expect_equal(
            errors,
            f"rounds[{round_label}].update_count",
            round_mapping.get("update_count"),
            expected_update_count,
        )


def verify_client_count_sweep_summary_path(
    summary_path: Path,
    *,
    expected_client_counts: tuple[int, ...],
    report_expectation: FederatedReportExpectation,
) -> VerificationResult:
    summary = _load_json_object(summary_path)
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
    _expect_equal(
        errors,
        "schema_version",
        summary.get("schema_version"),
        "fl_ssl_client_count_sweep_summary.v1",
    )
    observed_counts = tuple(
        int(value) for value in _object_sequence(summary.get("client_counts"))
    )
    if observed_counts != expected_client_counts:
        errors.append(
            "client_counts expected "
            f"{list(expected_client_counts)!r}, got {list(observed_counts)!r}."
        )
    _expect_equal(
        errors,
        "protocol.round_budget",
        _object_mapping(summary.get("protocol")).get("round_budget"),
        report_expectation.expected_round_budget,
    )
    runs = _object_sequence(summary.get("runs"))
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
    errors: list[str] = []
    for run in _object_sequence(summary.get("runs")):
        run_payload = _object_mapping(run)
        client_count = _optional_int(run_payload.get("client_count"))
        report_path = _resolve_report_path(summary_path, run_payload.get("report_path"))
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


def _load_json_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _object_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _object_sequence(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, list | tuple) else ()


def _nested_or_flat_value(
    payload: Mapping[str, object],
    namespace: str,
    key: str,
) -> object:
    flat_value = payload.get(f"{namespace}.{key}")
    if flat_value is not None:
        return flat_value
    return _object_mapping(payload.get(namespace)).get(key)


def _resolve_report_path(summary_path: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return summary_path.parent / path


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _expect_equal(
    errors: list[str],
    field: str,
    observed: object,
    expected: object,
) -> None:
    if expected is not None and observed != expected:
        errors.append(f"{field} expected {expected!r}, got {observed!r}.")


def _expect_float_equal(
    errors: list[str],
    field: str,
    observed: object,
    expected: float | None,
) -> None:
    if expected is None:
        return
    if observed is None:
        errors.append(f"{field} expected {expected!r}, got None.")
        return
    if float(observed) != expected:
        errors.append(f"{field} expected {expected!r}, got {observed!r}.")


def _expect_contains(
    errors: list[str],
    field: str,
    observed: object,
    expected_substring: str | None,
) -> None:
    if expected_substring is None:
        return
    if not isinstance(observed, str) or expected_substring not in observed:
        errors.append(
            f"{field} expected to contain {expected_substring!r}, got {observed!r}."
        )

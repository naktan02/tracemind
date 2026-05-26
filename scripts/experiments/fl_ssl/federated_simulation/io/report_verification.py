"""FL SSL report artifact verification helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from methods.adaptation.text_classifier.peft_encoder.report_artifacts import (
    classifier_aggregate_snapshot_candidates,
    classifier_objective_value,
    classifier_primary_update_ref_fields,
)

from .report_verification_helpers import (
    expect_contains as _expect_contains,
)
from .report_verification_helpers import (
    expect_equal as _expect_equal,
)
from .report_verification_helpers import (
    expect_float_equal as _expect_float_equal,
)
from .report_verification_helpers import (
    load_json_object as _load_json_object,
)
from .report_verification_helpers import (
    nested_or_flat_value as _nested_or_flat_value,
)
from .report_verification_helpers import (
    object_mapping as _object_mapping,
)
from .report_verification_helpers import (
    object_sequence as _object_sequence,
)
from .report_verification_helpers import (
    optional_int as _optional_int,
)
from .report_verification_models import (
    FederatedReportExpectation,
    VerificationResult,
)


def verify_federated_simulation_report_path(
    report_path: Path,
    expectation: FederatedReportExpectation,
) -> VerificationResult:
    try:
        payload = _load_json_object(report_path)
    except OSError as error:
        return VerificationResult(
            artifact=str(report_path),
            errors=(f"report file could not be read: {error}",),
        )
    except (json.JSONDecodeError, ValueError) as error:
        return VerificationResult(
            artifact=str(report_path),
            errors=(f"report file is not a valid JSON object: {error}",),
        )
    return verify_federated_simulation_report_payload(
        artifact=str(report_path),
        payload=payload,
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
    fl_capabilities = _object_mapping(protocol.get("fl_capabilities"))
    embedding_adapter = _object_mapping(
        protocol.get("embedding_adapter") or payload.get("embedding_adapter")
    )
    shard_policy = _object_mapping(protocol.get("shard_policy"))
    fl_data_source = _object_mapping(protocol.get("fl_data_source"))
    run_control = _object_mapping(protocol.get("run_control"))
    fl_method = _object_mapping(protocol.get("fl_method"))
    ssl_method = _object_mapping(protocol.get("ssl_method"))
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
    labeled_exposure_policy = _object_mapping(
        fl_data_source.get("labeled_exposure_policy")
    )
    _expect_equal(
        errors,
        "protocol.fl_data_source.labeled_exposure_policy.name",
        labeled_exposure_policy.get("name"),
        expectation.expected_labeled_exposure_policy,
    )
    _expect_equal(
        errors,
        "protocol.run_control.budget_name",
        run_control.get("budget_name"),
        expectation.expected_run_control_budget_name,
    )
    _expect_equal(
        errors,
        "protocol.run_control.output_dir",
        run_control.get("output_dir"),
        expectation.expected_run_control_output_dir,
    )
    _expect_equal(
        errors,
        "protocol.fl_method.name",
        fl_method.get("name"),
        expectation.expected_fl_method_name,
    )
    _expect_equal(
        errors,
        "protocol.fl_method.descriptor_name",
        fl_method.get("descriptor_name"),
        expectation.expected_fl_method_descriptor_name,
    )
    _expect_equal(
        errors,
        "protocol.fl_method.execution_role",
        fl_method.get("execution_role"),
        expectation.expected_fl_method_execution_role,
    )
    _expect_equal(
        errors,
        "protocol.ssl_method.name",
        ssl_method.get("name"),
        expectation.expected_federated_ssl_method,
    )
    _expect_equal(
        errors,
        "protocol.ssl_method.implementation_status",
        ssl_method.get("implementation_status"),
        expectation.expected_ssl_method_implementation_status,
    )
    _expect_equal(
        errors,
        "protocol.ssl_method.scenario",
        ssl_method.get("scenario"),
        expectation.expected_ssl_method_scenario,
    )
    _expect_equal(
        errors,
        "protocol.ssl_method.local_budget_policy",
        ssl_method.get("local_budget_policy"),
        expectation.expected_ssl_method_local_budget_policy,
    )
    _expect_equal(
        errors,
        "protocol.ssl_method.parameter_override_status",
        ssl_method.get("parameter_override_status"),
        expectation.expected_ssl_method_parameter_override_status,
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
    _expect_capability_policy(
        errors,
        fl_capabilities,
        "server_update_policy",
        expectation.expected_server_update_policy,
    )
    _expect_capability_policy(
        errors,
        fl_capabilities,
        "update_partition_policy",
        expectation.expected_update_partition_policy,
    )
    _expect_capability_policy(
        errors,
        fl_capabilities,
        "aggregation_weight_policy",
        expectation.expected_aggregation_weight_policy,
    )
    _expect_capability_policy(
        errors,
        fl_capabilities,
        "peer_context_policy",
        expectation.expected_peer_context_policy,
    )
    _expect_capability_policy(
        errors,
        fl_capabilities,
        "local_ssl_policy",
        expectation.expected_local_ssl_policy,
    )
    _expect_equal(
        errors,
        "objective.classifier.delta_format",
        classifier_objective_value(objective, "delta_format"),
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
    _verify_shared_update_artifacts(
        errors=errors,
        artifact=artifact,
        rounds=rounds,
        expectation=expectation,
    )
    _verify_posthoc_communication_cost(
        errors=errors,
        payload=payload,
        expectation=expectation,
    )
    return VerificationResult(artifact=artifact, errors=tuple(errors))


def _expect_capability_policy(
    errors: list[str],
    fl_capabilities: Mapping[str, object],
    policy_key: str,
    expected_name: str | None,
) -> None:
    policy = _object_mapping(fl_capabilities.get(policy_key))
    _expect_equal(
        errors,
        f"protocol.fl_capabilities.{policy_key}.name",
        policy.get("name"),
        expected_name,
    )


def verify_client_count_sweep_summary_path(
    summary_path: Path,
    *,
    expected_client_counts: tuple[int, ...],
    report_expectation: FederatedReportExpectation,
) -> VerificationResult:
    from .client_count_sweep_verification import (
        verify_client_count_sweep_summary_path as verify_summary_path,
    )

    return verify_summary_path(
        summary_path,
        expected_client_counts=expected_client_counts,
        report_expectation=report_expectation,
    )


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


def _verify_shared_update_artifacts(
    *,
    errors: list[str],
    artifact: str,
    rounds: tuple[object, ...],
    expectation: FederatedReportExpectation,
) -> None:
    if not _requires_shared_update_artifact_check(expectation):
        return
    report_path = Path(artifact)
    if not report_path.exists():
        errors.append(
            "report path must exist when shared update artifact checks are enabled: "
            f"{artifact}."
        )
        return
    run_dir = _run_dir_for_report_path(report_path)
    update_dir = run_dir / "main_server" / "shared_adapter_updates" / "versions"
    update_paths = sorted(update_dir.glob("*.json"))
    expected_count = _expected_shared_update_count(rounds, expectation, errors)
    _expect_equal(
        errors,
        "shared_adapter_updates.length",
        len(update_paths),
        expected_count,
    )
    if not update_paths:
        errors.append(f"shared update artifact directory is empty: {update_dir}.")
        return

    for update_path in update_paths:
        update_payload = _load_json_object(update_path)
        _verify_shared_update_payload(
            errors=errors,
            run_dir=run_dir,
            update_path=update_path,
            update_payload=update_payload,
            expectation=expectation,
        )
    if (
        expectation.expect_classifier_aggregate_snapshot
        or expectation.expect_peft_classifier_aggregate_snapshot
        or expectation.expect_lora_classifier_aggregate_snapshot
    ):
        _verify_classifier_aggregate_snapshot(
            errors=errors,
            run_dir=run_dir,
            rounds=rounds,
        )


def _verify_posthoc_communication_cost(
    *,
    errors: list[str],
    payload: Mapping[str, object],
    expectation: FederatedReportExpectation,
) -> None:
    if (
        expectation.expected_posthoc_communication_schema_version is None
        and not expectation.expect_partitioned_sparse_s2c_estimates
    ):
        return
    diagnostics = _object_mapping(payload.get("diagnostics"))
    diagnostic_cost = _object_mapping(diagnostics.get("communication_cost"))
    diagnostic_posthoc = _object_mapping(diagnostic_cost.get("posthoc_byte_estimates"))
    _expect_equal(
        errors,
        "diagnostics.communication_cost.posthoc_byte_estimates.schema_version",
        diagnostic_posthoc.get("schema_version"),
        expectation.expected_posthoc_communication_schema_version,
    )
    metrics = _object_mapping(payload.get("metrics"))
    secondary = _object_mapping(metrics.get("secondary"))
    secondary_cost = _object_mapping(secondary.get("communication_cost"))
    secondary_posthoc = _object_mapping(secondary_cost.get("posthoc_byte_estimates"))
    _expect_equal(
        errors,
        "metrics.secondary.communication_cost.posthoc_byte_estimates.schema_version",
        secondary_posthoc.get("schema_version"),
        expectation.expected_posthoc_communication_schema_version,
    )
    if not expectation.expect_partitioned_sparse_s2c_estimates:
        return
    _verify_partitioned_sparse_s2c_estimates(
        errors=errors,
        posthoc=diagnostic_posthoc,
        field_prefix="diagnostics.communication_cost.posthoc_byte_estimates",
    )
    _verify_partitioned_sparse_s2c_estimates(
        errors=errors,
        posthoc=secondary_posthoc,
        field_prefix="metrics.secondary.communication_cost.posthoc_byte_estimates",
    )


def _verify_partitioned_sparse_s2c_estimates(
    *,
    errors: list[str],
    posthoc: Mapping[str, object],
    field_prefix: str,
) -> None:
    if not posthoc:
        errors.append(f"{field_prefix} is required.")
        return
    if (
        _optional_int(posthoc.get("s2c_partitioned_sparse_transport_bytes_estimated"))
        is None
    ):
        errors.append(
            f"{field_prefix}.s2c_partitioned_sparse_transport_bytes_estimated "
            "is required."
        )
    per_round = _object_sequence(posthoc.get("per_round"))
    if not per_round:
        errors.append(f"{field_prefix}.per_round must not be empty.")
        return
    for index, round_payload in enumerate(per_round, start=1):
        round_mapping = _object_mapping(round_payload)
        if (
            _optional_int(
                round_mapping.get("s2c_partitioned_sparse_transport_bytes_estimated")
            )
            is None
        ):
            errors.append(
                f"{field_prefix}.per_round[{index}]"
                ".s2c_partitioned_sparse_transport_bytes_estimated is required."
            )


def _requires_shared_update_artifact_check(
    expectation: FederatedReportExpectation,
) -> bool:
    return any(
        (
            expectation.expected_shared_update_count is not None,
            expectation.expected_shared_update_count_matches_round_updates,
            expectation.expect_server_owned_update_artifacts,
            expectation.expect_partitioned_update_artifact_refs,
            expectation.expect_no_agent_local_update_refs,
            expectation.expect_classifier_aggregate_snapshot,
            expectation.expect_peft_classifier_aggregate_snapshot,
            expectation.expect_lora_classifier_aggregate_snapshot,
        )
    )


def _run_dir_for_report_path(report_path: Path) -> Path:
    if report_path.parent.name == "reports":
        return report_path.parent.parent
    return report_path.parent


def _expected_shared_update_count(
    rounds: tuple[object, ...],
    expectation: FederatedReportExpectation,
    errors: list[str],
) -> int | None:
    if expectation.expected_shared_update_count is not None:
        return expectation.expected_shared_update_count
    if not expectation.expected_shared_update_count_matches_round_updates:
        return None
    update_counts: list[int] = []
    for index, round_payload in enumerate(rounds, start=1):
        round_mapping = _object_mapping(round_payload)
        update_count = _optional_int(round_mapping.get("update_count"))
        if update_count is None:
            errors.append(
                "round update_count is required when shared update count must "
                f"match round updates: round_index={index}."
            )
            return None
        update_counts.append(update_count)
    return sum(update_counts)


def _verify_shared_update_payload(
    *,
    errors: list[str],
    run_dir: Path,
    update_path: Path,
    update_payload: Mapping[str, object],
    expectation: FederatedReportExpectation,
) -> None:
    update_label = str(update_path)
    _expect_equal(
        errors,
        f"{update_label}.delta_format",
        update_payload.get("delta_format"),
        expectation.expected_delta_format,
    )
    if expectation.expect_no_agent_local_update_refs and (
        "agent-local://" in json.dumps(update_payload, sort_keys=True)
    ):
        errors.append(f"{update_label} must not contain agent-local artifact refs.")
    if expectation.expect_partitioned_update_artifact_refs:
        _verify_partitioned_update_ref(
            errors=errors,
            run_dir=run_dir,
            update_label=update_label,
            update_payload=update_payload,
        )
        return
    if not expectation.expect_server_owned_update_artifacts:
        return
    _verify_server_owned_update_refs(
        errors=errors,
        run_dir=run_dir,
        update_label=update_label,
        update_payload=update_payload,
    )


def _verify_partitioned_update_ref(
    *,
    errors: list[str],
    run_dir: Path,
    update_label: str,
    update_payload: Mapping[str, object],
) -> None:
    partitioned_ref = update_payload.get("partitioned_deltas_artifact_ref")
    if partitioned_ref is None:
        errors.append(
            f"{update_label}.partitioned_deltas_artifact_ref is required for "
            "partitioned update verification."
        )
        return
    _verify_server_owned_artifact_ref(
        errors=errors,
        run_dir=run_dir,
        update_label=update_label,
        field_name="partitioned_deltas_artifact_ref",
        artifact_ref=partitioned_ref,
    )


def _verify_server_owned_update_refs(
    *,
    errors: list[str],
    run_dir: Path,
    update_label: str,
    update_payload: Mapping[str, object],
) -> None:
    """merged-delta와 partitioned-only update artifact ref를 함께 검증한다."""

    partitioned_ref = update_payload.get("partitioned_deltas_artifact_ref")
    if partitioned_ref is not None:
        _verify_server_owned_artifact_ref(
            errors=errors,
            run_dir=run_dir,
            update_label=update_label,
            field_name="partitioned_deltas_artifact_ref",
            artifact_ref=partitioned_ref,
        )
        return
    primary_ref_fields = classifier_primary_update_ref_fields(update_payload)
    for field_name, artifact_ref in zip(
        primary_ref_fields,
        (update_payload.get(field_name) for field_name in primary_ref_fields),
        strict=True,
    ):
        _verify_server_owned_artifact_ref(
            errors=errors,
            run_dir=run_dir,
            update_label=update_label,
            field_name=field_name,
            artifact_ref=artifact_ref,
        )


def _verify_server_owned_artifact_ref(
    *,
    errors: list[str],
    run_dir: Path,
    update_label: str,
    field_name: str,
    artifact_ref: object,
) -> None:
    if not isinstance(artifact_ref, str) or not artifact_ref:
        errors.append(f"{update_label}.{field_name} must be a non-empty string.")
        return
    if not artifact_ref.startswith("aggregation_artifact::"):
        errors.append(
            f"{update_label}.{field_name} must start with "
            f"'aggregation_artifact::', got {artifact_ref!r}."
        )
        return
    artifact_path = _aggregation_artifact_path(run_dir, artifact_ref)
    if not artifact_path.exists():
        errors.append(
            f"{update_label}.{field_name} target does not exist: {artifact_path}."
        )


def _aggregation_artifact_path(run_dir: Path, artifact_ref: str) -> Path:
    relative_ref = artifact_ref.removeprefix("aggregation_artifact::")
    artifact_path = (
        run_dir / "main_server" / "aggregation_artifacts" / "versions" / relative_ref
    )
    if artifact_path.suffix:
        return artifact_path
    json_path = artifact_path.with_suffix(".json")
    if json_path.exists():
        return json_path
    safetensors_path = artifact_path.with_suffix(".safetensors")
    if safetensors_path.exists():
        return safetensors_path
    return json_path


def _verify_classifier_aggregate_snapshot(
    *,
    errors: list[str],
    run_dir: Path,
    rounds: tuple[object, ...],
) -> None:
    if not rounds:
        errors.append(
            "rounds must not be empty when peft classifier aggregate snapshot "
            "is expected."
        )
        return
    final_round = _object_mapping(rounds[-1])
    model_revision = final_round.get("model_revision")
    if not isinstance(model_revision, str) or not model_revision:
        errors.append(
            "rounds[-1].model_revision is required when peft classifier aggregate "
            "snapshot is expected."
        )
        return
    artifact_root = run_dir / "main_server" / "aggregation_artifacts" / "versions"
    candidate_paths = classifier_aggregate_snapshot_candidates(
        artifact_root=artifact_root,
        model_revision=model_revision,
    )
    if any(all(path.exists() for path in candidate) for candidate in candidate_paths):
        return
    for candidate in candidate_paths:
        for artifact_path in candidate:
            if not artifact_path.exists():
                errors.append(
                    "peft classifier aggregate snapshot artifact does not exist: "
                    f"{artifact_path}."
                )

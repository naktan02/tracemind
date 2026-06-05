"""FL SSL report artifact verifier tests."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.fl_ssl.federated_simulation.io import (
    communication_cost_estimates,
)
from scripts.experiments.fl_ssl.federated_simulation.io.report_verification import (
    FederatedReportExpectation,
    verify_client_count_sweep_summary_path,
    verify_federated_simulation_report_path,
    verify_federated_simulation_report_payload,
)
from scripts.experiments.fl_ssl.verify_federated_report_artifacts import (
    _optional_manifest_path,
)
from scripts.experiments.fl_ssl.verify_federated_report_artifacts import (
    main as verify_federated_report_artifacts_main,
)


def _report_payload(
    *,
    client_count: int,
    completed_rounds: int,
    round_budget: int,
    seed: int = 42,
    shard_alpha: float = 0.3,
    split_id: str = "example_alpha0.3_clients2_seed42",
    labeled_exposure_policy: str = "client_local_split",
    run_control_budget_name: str = "main",
    run_control_output_dir: str = "runs/fl_ssl",
    fl_method_name: str = "manual",
    fl_method_descriptor_name: str | None = None,
    fl_method_execution_role: str = "manual_baseline",
    federated_ssl_method: str | None = None,
    ssl_method_implementation_status: str | None = None,
    ssl_method_scenario: str | None = None,
    ssl_method_local_budget_policy: str | None = None,
    ssl_method_parameter_override_status: str | None = None,
    ssl_algorithm: str = "fixmatch",
    ssl_method: str = "fixmatch_usb_v1",
    fallback_payload_adapter_kind: str = "peft_classifier",
    update_family: str = "peft_text_encoder",
    aggregation: str = "fedavg",
    server_update_policy: str = "fedavg_merged_delta",
    update_partition_policy: str = "unified",
    aggregation_weight_policy: str = "example_count",
    peer_context_policy: str = "none",
    local_ssl_policy: str = "query_ssl_method",
    delta_format: str = "server_uploaded_artifact_ref",
    objective_payload_scope: str = "peft_classifier",
    payload_adapter_kind: str | None = None,
    embedding_backend: str = "transformers_mxbai",
    embedding_model_id: str = "mixedbread-ai/mxbai-embed-large-v1",
    embedding_device: str = "cuda",
    embedding_local_files_only: bool = True,
    local_trainer_device: str = "cuda",
    local_trainer_local_files_only: bool = True,
    round_update_count: int | None = None,
) -> dict[str, object]:
    effective_round_update_count = (
        client_count if round_update_count is None else round_update_count
    )
    return {
        "schema_version": "fl_ssl_main_comparison.v1",
        "protocol": {
            "client_count": client_count,
            "completed_rounds": completed_rounds,
            "round_budget": round_budget,
            "seed": seed,
            "shard_policy": {
                "name": "dirichlet_label_skew",
                "alpha": shard_alpha,
            },
            "fl_data_source": {
                "split_id": split_id,
                "labeled_exposure_policy": {"name": labeled_exposure_policy},
            },
            "run_control": {
                "metadata_status": "recorded",
                "budget_name": run_control_budget_name,
                "output_dir": run_control_output_dir,
            },
            "fl_method": {
                "name": fl_method_name,
                "descriptor_name": fl_method_descriptor_name,
                "execution_role": fl_method_execution_role,
            },
            "ssl_method": (
                None
                if federated_ssl_method is None
                else {
                    "name": federated_ssl_method,
                    "implementation_status": ssl_method_implementation_status,
                    "scenario": ssl_method_scenario,
                    "local_budget_policy": ssl_method_local_budget_policy,
                    "parameter_override_status": ssl_method_parameter_override_status,
                }
            ),
            "objective": {
                "query_ssl.algorithm_name": ssl_algorithm,
                "query_ssl.method_name": ssl_method,
                f"{objective_payload_scope}.delta_format": delta_format,
            },
            "round_runtime": {
                "payload_adapter_kind": (
                    payload_adapter_kind or fallback_payload_adapter_kind
                ),
                "update_family_name": update_family,
                "aggregation_backend_name": aggregation,
            },
            "fl_capabilities": {
                "metadata_status": "recorded",
                "server_update_policy": {"name": server_update_policy},
                "update_partition_policy": {"name": update_partition_policy},
                "aggregation_weight_policy": {"name": aggregation_weight_policy},
                "peer_context_policy": {"name": peer_context_policy},
                "local_ssl_policy": {"name": local_ssl_policy},
            },
            "embedding_adapter": {
                "metadata_status": "recorded",
                "backend": embedding_backend,
                "model_id": embedding_model_id,
                "device": embedding_device,
                "local_files_only": embedding_local_files_only,
            },
            "local_trainer_runtime": {
                "metadata_status": "recorded",
                "device": local_trainer_device,
                "local_files_only": local_trainer_local_files_only,
            },
        },
        "rounds": [
            {
                "round_id": f"round_{round_index:04d}",
                "round_index": round_index,
                "model_revision": f"sim_rev_{round_index:04d}",
                "update_count": effective_round_update_count,
            }
            for round_index in range(1, completed_rounds + 1)
        ],
    }


def _expectation(
    *,
    completed_rounds: int = 1,
    round_budget: int = 1,
    client_count: int | None = 2,
) -> FederatedReportExpectation:
    return FederatedReportExpectation(
        expected_completed_rounds=completed_rounds,
        expected_round_budget=round_budget,
        expected_client_count=client_count,
        expected_seed=42,
        expected_shard_policy_name="dirichlet_label_skew",
        expected_shard_alpha=0.3,
        expected_split_id_contains="alpha0.3",
        expected_labeled_exposure_policy="client_local_split",
        expected_run_control_budget_name="main",
        expected_run_control_output_dir="runs/fl_ssl",
        expected_ssl_algorithm="fixmatch",
        expected_ssl_method="fixmatch_usb_v1",
        expected_payload_adapter_kind="peft_classifier",
        expected_update_family="peft_text_encoder",
        expected_aggregation="fedavg",
        expected_server_update_policy="fedavg_merged_delta",
        expected_update_partition_policy="unified",
        expected_aggregation_weight_policy="example_count",
        expected_peer_context_policy="none",
        expected_local_ssl_policy="query_ssl_method",
        expected_delta_format="server_uploaded_artifact_ref",
    )


def _runtime_metadata_expectation() -> FederatedReportExpectation:
    return FederatedReportExpectation(
        expected_embedding_metadata_status="recorded",
        expected_embedding_backend="transformers_mxbai",
        expected_embedding_model_id="mixedbread-ai/mxbai-embed-large-v1",
        expected_embedding_device="cuda",
        expected_embedding_local_files_only=True,
        expected_local_trainer_metadata_status="recorded",
        expected_local_trainer_device="cuda",
        expected_local_trainer_local_files_only=True,
    )


def _round_record_expectation() -> FederatedReportExpectation:
    return FederatedReportExpectation(
        expected_round_record_count=2,
        expected_round_update_count_matches_client_count=True,
    )


def _attach_artifact_communication_estimate(
    payload: dict[str, object],
    *,
    include_sparse_estimate: bool = True,
    mirror_secondary: bool = True,
) -> dict[str, object]:
    estimate: dict[str, object] = {
        "schema_version": (
            communication_cost_estimates.COMMUNICATION_ESTIMATE_SCHEMA_VERSION
        ),
        "c2s_total_bytes": 28,
        "s2c_total_bytes_estimated": 336,
        "bidirectional_total_bytes_estimated": 364,
        "per_round": [
            {
                "round_id": "round_0001",
                "s2c_partitioned_sparse_transport_bytes_estimated": 0,
            },
            {
                "round_id": "round_0002",
                "s2c_partitioned_sparse_transport_bytes_estimated": 96,
            },
        ],
    }
    if include_sparse_estimate:
        estimate["s2c_partitioned_sparse_transport_bytes_estimated"] = 96
    diagnostics = {"communication_cost": {"artifact_byte_estimates": estimate}}
    metrics = {
        "secondary": {"communication_cost": {"artifact_byte_estimates": estimate}}
    }
    payload["diagnostics"] = diagnostics
    payload["metrics"] = metrics if mirror_secondary else {"secondary": {}}
    return payload


def _write_report_run_with_server_update_artifacts(
    tmp_path: Path,
    *,
    payload: dict[str, object],
    partitioned_only: bool = False,
) -> Path:
    run_dir = tmp_path / "run"
    report_path = run_dir / "reports" / "fl_ssl_main_comparison.report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    update_dir = run_dir / "main_server" / "shared_adapter_updates" / "versions"
    artifact_root = run_dir / "main_server" / "aggregation_artifacts" / "versions"
    update_dir.mkdir(parents=True)
    for round_payload in payload["rounds"]:
        round_mapping = dict(round_payload)  # type: ignore[arg-type]
        round_id = str(round_mapping["round_id"])
        update_count = int(round_mapping["update_count"])
        for client_index in range(1, update_count + 1):
            client_id = f"agent_{client_index:02d}"
            update_id = f"update_{round_id}_{client_id}"
            ref_prefix = f"client_updates/{round_id}/{client_id}/{update_id}"
            adapter_delta_name = "peft_adapter_delta"
            adapter_ref_field = "peft_adapter_delta_artifact_ref"
            adapter_ref = f"aggregation_artifact::{ref_prefix}/{adapter_delta_name}"
            head_ref = f"aggregation_artifact::{ref_prefix}/classifier_head_delta"
            partitioned_ref = f"aggregation_artifact::{ref_prefix}/partitioned_delta"
            (artifact_root / ref_prefix).mkdir(parents=True)
            if partitioned_only:
                partitioned_path = (
                    artifact_root / f"{ref_prefix}/partitioned_delta.safetensors"
                )
                partitioned_path.write_text("fake-safetensors", encoding="utf-8")
            else:
                (artifact_root / f"{ref_prefix}/{adapter_delta_name}.json").write_text(
                    json.dumps({"peft_parameters": {"a": [1.0]}}),
                    encoding="utf-8",
                )
                (artifact_root / f"{ref_prefix}/classifier_head_delta.json").write_text(
                    json.dumps({"classifier_head_weights": {"normal": [1.0]}}),
                    encoding="utf-8",
                )
            update_payload = {
                "update_id": update_id,
                "delta_format": "server_uploaded_artifact_ref",
            }
            if partitioned_only:
                update_payload["partitioned_deltas_artifact_ref"] = partitioned_ref
            else:
                update_payload[adapter_ref_field] = adapter_ref
                update_payload["classifier_head_delta_artifact_ref"] = head_ref
            (update_dir / f"{update_id}.json").write_text(
                json.dumps(update_payload),
                encoding="utf-8",
            )

    final_round = dict(payload["rounds"][-1])  # type: ignore[index,arg-type]
    snapshot_family = "peft_text_encoder"
    adapter_snapshot_name = "peft_adapter.safetensors"
    snapshot_dir = artifact_root / snapshot_family / str(final_round["model_revision"])
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / adapter_snapshot_name).write_bytes(b"placeholder")
    (snapshot_dir / "classifier_head.json").write_text(
        json.dumps({"classifier_head_weights": {"normal": [1.0]}}),
        encoding="utf-8",
    )
    return report_path


def test_verify_federated_report_accepts_expected_runtime_metadata() -> None:
    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=_report_payload(client_count=2, completed_rounds=1, round_budget=1),
        expectation=_expectation(),
    )

    assert result.passed
    assert result.errors == ()


def test_verify_federated_report_flags_method_and_runtime_drift() -> None:
    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=_report_payload(
            client_count=2,
            completed_rounds=1,
            round_budget=1,
            ssl_algorithm="pseudolabel",
            payload_adapter_kind="diagonal_scale",
        ),
        expectation=_expectation(),
    )

    assert not result.passed
    assert (
        "objective.query_ssl.algorithm_name expected 'fixmatch', got 'pseudolabel'."
        in result.errors
    )
    assert (
        "round_runtime.payload_adapter_kind expected 'peft_classifier', "
        "got 'diagonal_scale'." in result.errors
    )


def test_verify_federated_report_uses_payload_adapter_kind() -> None:
    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=_report_payload(
            client_count=2,
            completed_rounds=1,
            round_budget=1,
            fallback_payload_adapter_kind="peft_classifier",
            payload_adapter_kind="peft_classifier",
        ),
        expectation=FederatedReportExpectation(
            expected_payload_adapter_kind="peft_classifier",
        ),
    )

    assert result.passed


def test_verify_federated_report_flags_update_family_drift() -> None:
    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=_report_payload(
            client_count=2,
            completed_rounds=1,
            round_budget=1,
            update_family="prototype_pack",
        ),
        expectation=_expectation(),
    )

    assert not result.passed
    assert (
        "round_runtime.update_family_name expected 'peft_text_encoder', "
        "got 'prototype_pack'." in result.errors
    )


def test_verify_federated_report_flags_split_condition_drift() -> None:
    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=_report_payload(
            client_count=2,
            completed_rounds=1,
            round_budget=1,
            seed=43,
            shard_alpha=0.1,
            split_id="example_alpha0.1_clients2_seed43",
        ),
        expectation=FederatedReportExpectation(
            expected_seed=42,
            expected_shard_policy_name="dirichlet_label_skew",
            expected_shard_alpha=0.3,
            expected_split_id_contains="alpha0.3",
        ),
    )

    assert not result.passed
    assert "protocol.seed expected 42, got 43." in result.errors
    assert "protocol.shard_policy.alpha expected 0.3, got 0.1." in result.errors
    assert (
        "protocol.fl_data_source.split_id expected to contain 'alpha0.3', "
        "got 'example_alpha0.1_clients2_seed43'." in result.errors
    )


def test_verify_federated_report_flags_labeled_exposure_policy_drift() -> None:
    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=_report_payload(
            client_count=2,
            completed_rounds=1,
            round_budget=1,
            labeled_exposure_policy="client_local_split",
        ),
        expectation=FederatedReportExpectation(
            expected_labeled_exposure_policy="shared_client_seed",
        ),
    )

    assert not result.passed
    assert (
        "protocol.fl_data_source.labeled_exposure_policy.name expected "
        "'shared_client_seed', got 'client_local_split'."
    ) in result.errors


def test_verify_federated_report_flags_run_control_drift() -> None:
    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=_report_payload(
            client_count=2,
            completed_rounds=1,
            round_budget=1,
            run_control_budget_name="smoke",
            run_control_output_dir="runs/_smoke/fl_ssl",
        ),
        expectation=FederatedReportExpectation(
            expected_run_control_budget_name="reduced",
            expected_run_control_output_dir="runs/fl_ssl",
        ),
    )

    assert not result.passed
    assert (
        "protocol.run_control.budget_name expected 'reduced', got 'smoke'."
        in result.errors
    )
    assert (
        "protocol.run_control.output_dir expected 'runs/fl_ssl', "
        "got 'runs/_smoke/fl_ssl'."
    ) in result.errors


def test_verify_federated_report_path_returns_failure_for_missing_report(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "missing.report.json"

    result = verify_federated_simulation_report_path(
        report_path,
        FederatedReportExpectation(),
    )

    assert not result.passed
    assert result.artifact == str(report_path)
    assert result.errors
    assert result.errors[0].startswith("report file could not be read:")


def test_verify_federated_report_accepts_embedding_and_trainer_metadata() -> None:
    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=_report_payload(client_count=2, completed_rounds=1, round_budget=1),
        expectation=_runtime_metadata_expectation(),
    )

    assert result.passed


def test_verify_federated_report_flags_gpu_mxbai_metadata_drift() -> None:
    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=_report_payload(
            client_count=2,
            completed_rounds=1,
            round_budget=1,
            embedding_backend="hash_debug",
            embedding_device="cpu",
            local_trainer_local_files_only=False,
        ),
        expectation=_runtime_metadata_expectation(),
    )

    assert not result.passed
    assert (
        "embedding_adapter.backend expected 'transformers_mxbai', "
        "got 'hash_debug'." in result.errors
    )
    assert "embedding_adapter.device expected 'cuda', got 'cpu'." in result.errors
    assert (
        "local_trainer_runtime.local_files_only expected True, got False."
        in result.errors
    )


def test_verify_federated_report_accepts_round_records_and_update_counts() -> None:
    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=_report_payload(client_count=2, completed_rounds=2, round_budget=2),
        expectation=_round_record_expectation(),
    )

    assert result.passed


def test_verify_federated_report_flags_round_record_drift() -> None:
    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=_report_payload(
            client_count=2,
            completed_rounds=1,
            round_budget=2,
            round_update_count=1,
        ),
        expectation=_round_record_expectation(),
    )

    assert not result.passed
    assert "rounds.length expected 2, got 1." in result.errors
    assert "rounds[round_0001].update_count expected 2, got 1." in result.errors


def test_verify_federated_report_checks_server_owned_update_artifacts(
    tmp_path: Path,
) -> None:
    report_path = _write_report_run_with_server_update_artifacts(
        tmp_path,
        payload=_report_payload(client_count=2, completed_rounds=2, round_budget=2),
    )

    result = verify_federated_simulation_report_path(
        report_path,
        FederatedReportExpectation(
            expected_delta_format="server_uploaded_artifact_ref",
            expected_shared_update_count_matches_round_updates=True,
            expect_server_owned_update_artifacts=True,
            expect_no_agent_local_update_refs=True,
            expect_peft_encoder_aggregate_snapshot=True,
        ),
    )

    assert result.passed


def test_verify_federated_report_accepts_peft_classifier_v2_update_artifacts(
    tmp_path: Path,
) -> None:
    report_path = _write_report_run_with_server_update_artifacts(
        tmp_path,
        payload=_report_payload(
            client_count=2,
            completed_rounds=2,
            round_budget=2,
            fallback_payload_adapter_kind="peft_classifier",
            objective_payload_scope="peft_classifier",
        ),
    )

    result = verify_federated_simulation_report_path(
        report_path,
        FederatedReportExpectation(
            expected_payload_adapter_kind="peft_classifier",
            expected_delta_format="server_uploaded_artifact_ref",
            expected_shared_update_count_matches_round_updates=True,
            expect_server_owned_update_artifacts=True,
            expect_no_agent_local_update_refs=True,
            expect_peft_encoder_aggregate_snapshot=True,
        ),
    )

    assert result.passed


def test_update_family_snapshot_expectation_is_canonical() -> None:
    expectation = FederatedReportExpectation(
        expected_aggregate_snapshot_update_family="peft_text_encoder",
    )

    assert expectation.aggregate_snapshot_update_family == "peft_text_encoder"


def test_peft_encoder_snapshot_expectation_stays_compatibility_alias() -> None:
    expectation = FederatedReportExpectation(
        expect_peft_encoder_aggregate_snapshot=True,
    )

    assert expectation.aggregate_snapshot_update_family == "peft_text_encoder"


def test_verify_federated_report_accepts_partitioned_only_update_artifacts(
    tmp_path: Path,
) -> None:
    report_path = _write_report_run_with_server_update_artifacts(
        tmp_path,
        payload=_report_payload(client_count=2, completed_rounds=1, round_budget=1),
        partitioned_only=True,
    )

    result = verify_federated_simulation_report_path(
        report_path,
        FederatedReportExpectation(
            expected_delta_format="server_uploaded_artifact_ref",
            expected_shared_update_count_matches_round_updates=True,
            expect_server_owned_update_artifacts=True,
            expect_no_agent_local_update_refs=True,
            expect_peft_encoder_aggregate_snapshot=True,
        ),
    )

    assert result.passed


def test_verify_fedmatch_partitioned_report_requires_capabilities_and_partition_ref(
    tmp_path: Path,
) -> None:
    report_path = _write_report_run_with_server_update_artifacts(
        tmp_path,
        payload=_report_payload(
            client_count=2,
            completed_rounds=1,
            round_budget=1,
            fl_method_name="fedmatch",
            fl_method_descriptor_name="fedmatch",
            fl_method_execution_role="method_owned",
            federated_ssl_method="fedmatch",
            ssl_method_implementation_status="partitioned_trainable_state_slice_v1",
            ssl_method_scenario="labels-at-client",
            ssl_method_local_budget_policy="iteration_capped",
            ssl_method_parameter_override_status="original",
            server_update_policy="fedmatch_partitioned",
            update_partition_policy="partitioned",
            aggregation_weight_policy="uniform",
            peer_context_policy="fixed_probe_output_knn",
            local_ssl_policy="fedmatch_agreement",
        ),
        partitioned_only=True,
    )

    result = verify_federated_simulation_report_path(
        report_path,
        FederatedReportExpectation(
            expected_fl_method_name="fedmatch",
            expected_fl_method_descriptor_name="fedmatch",
            expected_fl_method_execution_role="method_owned",
            expected_federated_ssl_method="fedmatch",
            expected_ssl_method_implementation_status=(
                "partitioned_trainable_state_slice_v1"
            ),
            expected_ssl_method_scenario="labels-at-client",
            expected_ssl_method_local_budget_policy="iteration_capped",
            expected_ssl_method_parameter_override_status="original",
            expected_server_update_policy="fedmatch_partitioned",
            expected_update_partition_policy="partitioned",
            expected_aggregation_weight_policy="uniform",
            expected_peer_context_policy="fixed_probe_output_knn",
            expected_local_ssl_policy="fedmatch_agreement",
            expected_delta_format="server_uploaded_artifact_ref",
            expected_shared_update_count_matches_round_updates=True,
            expect_partitioned_update_artifact_refs=True,
            expect_no_agent_local_update_refs=True,
        ),
    )

    assert result.passed


def test_verify_fedmatch_partitioned_report_rejects_primary_only_update_refs(
    tmp_path: Path,
) -> None:
    report_path = _write_report_run_with_server_update_artifacts(
        tmp_path,
        payload=_report_payload(
            client_count=1,
            completed_rounds=1,
            round_budget=1,
            fl_method_name="fedmatch",
            fl_method_descriptor_name="fedmatch",
            fl_method_execution_role="method_owned",
            federated_ssl_method="fedmatch",
            server_update_policy="fedmatch_partitioned",
            update_partition_policy="partitioned",
            aggregation_weight_policy="uniform",
            peer_context_policy="fixed_probe_output_knn",
            local_ssl_policy="fedmatch_agreement",
        ),
        partitioned_only=False,
    )

    result = verify_federated_simulation_report_path(
        report_path,
        FederatedReportExpectation(
            expected_fl_method_name="fedmatch",
            expected_fl_method_descriptor_name="fedmatch",
            expected_fl_method_execution_role="method_owned",
            expected_federated_ssl_method="fedmatch",
            expected_server_update_policy="fedmatch_partitioned",
            expected_update_partition_policy="partitioned",
            expected_aggregation_weight_policy="uniform",
            expected_peer_context_policy="fixed_probe_output_knn",
            expected_local_ssl_policy="fedmatch_agreement",
            expected_delta_format="server_uploaded_artifact_ref",
            expected_shared_update_count_matches_round_updates=True,
            expect_partitioned_update_artifact_refs=True,
        ),
    )

    assert not result.passed
    assert any(
        ".partitioned_deltas_artifact_ref is required for partitioned update "
        "verification." in error
        for error in result.errors
    )


def test_verify_fedmatch_partitioned_report_flags_capability_drift() -> None:
    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=_report_payload(
            client_count=2,
            completed_rounds=1,
            round_budget=1,
            fl_method_name="fedmatch",
            fl_method_descriptor_name="fedmatch",
            fl_method_execution_role="manual_baseline",
            federated_ssl_method="fedmatch",
            server_update_policy="fedavg_merged_delta",
            update_partition_policy="unified",
        ),
        expectation=FederatedReportExpectation(
            expected_fl_method_name="fedmatch",
            expected_fl_method_descriptor_name="fedmatch",
            expected_fl_method_execution_role="method_owned",
            expected_federated_ssl_method="fedmatch",
            expected_server_update_policy="fedmatch_partitioned",
            expected_update_partition_policy="partitioned",
        ),
    )

    assert not result.passed
    assert (
        "protocol.fl_capabilities.server_update_policy.name expected "
        "'fedmatch_partitioned', got 'fedavg_merged_delta'." in result.errors
    )
    assert (
        "protocol.fl_capabilities.update_partition_policy.name expected "
        "'partitioned', got 'unified'." in result.errors
    )
    assert (
        "protocol.fl_method.execution_role expected 'method_owned', "
        "got 'manual_baseline'." in result.errors
    )


def test_verify_fedmatch_partitioned_report_flags_ssl_method_protocol_drift() -> None:
    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=_report_payload(
            client_count=2,
            completed_rounds=1,
            round_budget=1,
            fl_method_name="fedmatch",
            fl_method_descriptor_name="fedmatch",
            fl_method_execution_role="method_owned",
            federated_ssl_method="fedmatch",
            ssl_method_implementation_status="metadata_only",
            ssl_method_scenario="labels-at-server",
            ssl_method_local_budget_policy="original_method",
            ssl_method_parameter_override_status="ablation",
        ),
        expectation=FederatedReportExpectation(
            expected_federated_ssl_method="fedmatch",
            expected_ssl_method_implementation_status=(
                "partitioned_trainable_state_slice_v1"
            ),
            expected_ssl_method_scenario="labels-at-client",
            expected_ssl_method_local_budget_policy="iteration_capped",
            expected_ssl_method_parameter_override_status="original",
        ),
    )

    assert not result.passed
    assert (
        "protocol.ssl_method.implementation_status expected "
        "'partitioned_trainable_state_slice_v1', got 'metadata_only'." in result.errors
    )
    assert (
        "protocol.ssl_method.scenario expected 'labels-at-client', "
        "got 'labels-at-server'." in result.errors
    )
    assert (
        "protocol.ssl_method.local_budget_policy expected 'iteration_capped', "
        "got 'original_method'." in result.errors
    )
    assert (
        "protocol.ssl_method.parameter_override_status expected 'original', "
        "got 'ablation'." in result.errors
    )


def test_verify_fedmatch_report_accepts_artifact_sparse_communication_cost() -> None:
    payload = _attach_artifact_communication_estimate(
        _report_payload(
            client_count=2,
            completed_rounds=2,
            round_budget=2,
            federated_ssl_method="fedmatch",
        )
    )

    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=payload,
        expectation=FederatedReportExpectation(
            expected_federated_ssl_method="fedmatch",
            expected_communication_estimate_schema_version=(
                communication_cost_estimates.COMMUNICATION_ESTIMATE_SCHEMA_VERSION
            ),
            expect_partitioned_sparse_s2c_estimates=True,
        ),
    )

    assert result.passed


def test_verify_fedmatch_report_flags_missing_artifact_sparse_cost_fields() -> None:
    payload = _attach_artifact_communication_estimate(
        _report_payload(
            client_count=2,
            completed_rounds=2,
            round_budget=2,
            federated_ssl_method="fedmatch",
        ),
        include_sparse_estimate=False,
        mirror_secondary=False,
    )

    result = verify_federated_simulation_report_payload(
        artifact="report.json",
        payload=payload,
        expectation=FederatedReportExpectation(
            expected_federated_ssl_method="fedmatch",
            expected_communication_estimate_schema_version=(
                communication_cost_estimates.COMMUNICATION_ESTIMATE_SCHEMA_VERSION
            ),
            expect_partitioned_sparse_s2c_estimates=True,
        ),
    )

    assert not result.passed
    assert (
        "diagnostics.communication_cost.artifact_byte_estimates"
        ".s2c_partitioned_sparse_transport_bytes_estimated is required."
    ) in result.errors
    assert (
        "metrics.secondary.communication_cost.artifact_byte_estimates.schema_version "
        "expected "
        f"'{communication_cost_estimates.COMMUNICATION_ESTIMATE_SCHEMA_VERSION}', "
        "got None."
    ) in result.errors
    assert (
        "metrics.secondary.communication_cost.artifact_byte_estimates is required."
        in (result.errors)
    )


def test_verify_federated_report_flags_agent_local_update_artifact_drift(
    tmp_path: Path,
) -> None:
    report_path = _write_report_run_with_server_update_artifacts(
        tmp_path,
        payload=_report_payload(client_count=1, completed_rounds=1, round_budget=1),
    )
    update_path = next(
        (
            report_path.parent.parent
            / "main_server"
            / "shared_adapter_updates"
            / "versions"
        ).glob("*.json")
    )
    update_payload = json.loads(update_path.read_text(encoding="utf-8"))
    update_payload["peft_adapter_delta_artifact_ref"] = (
        "agent-local://agent_01/peft_adapter_delta"
    )
    update_path.write_text(json.dumps(update_payload), encoding="utf-8")

    result = verify_federated_simulation_report_path(
        report_path,
        FederatedReportExpectation(
            expected_delta_format="server_uploaded_artifact_ref",
            expected_shared_update_count_matches_round_updates=True,
            expect_server_owned_update_artifacts=True,
            expect_no_agent_local_update_refs=True,
            expect_peft_encoder_aggregate_snapshot=True,
        ),
    )

    assert not result.passed
    assert any(
        error.endswith("must not contain agent-local artifact refs.")
        for error in result.errors
    )
    assert any(
        ".peft_adapter_delta_artifact_ref must start with 'aggregation_artifact::'"
        in error
        for error in result.errors
    )


def test_verify_federated_report_flags_peft_classifier_v2_artifact_ref_drift(
    tmp_path: Path,
) -> None:
    report_path = _write_report_run_with_server_update_artifacts(
        tmp_path,
        payload=_report_payload(
            client_count=1,
            completed_rounds=1,
            round_budget=1,
            fallback_payload_adapter_kind="peft_classifier",
            objective_payload_scope="peft_classifier",
        ),
    )
    update_path = next(
        (
            report_path.parent.parent
            / "main_server"
            / "shared_adapter_updates"
            / "versions"
        ).glob("*.json")
    )
    update_payload = json.loads(update_path.read_text(encoding="utf-8"))
    update_payload["peft_adapter_delta_artifact_ref"] = (
        "agent-local://agent_01/peft_adapter_delta"
    )
    update_path.write_text(json.dumps(update_payload), encoding="utf-8")

    result = verify_federated_simulation_report_path(
        report_path,
        FederatedReportExpectation(
            expected_payload_adapter_kind="peft_classifier",
            expected_delta_format="server_uploaded_artifact_ref",
            expected_shared_update_count_matches_round_updates=True,
            expect_server_owned_update_artifacts=True,
            expect_no_agent_local_update_refs=True,
            expect_peft_encoder_aggregate_snapshot=True,
        ),
    )

    assert not result.passed
    assert any(
        error.endswith("must not contain agent-local artifact refs.")
        for error in result.errors
    )
    assert any(
        ".peft_adapter_delta_artifact_ref must start with 'aggregation_artifact::'"
        in error
        for error in result.errors
    )


def test_verify_client_count_sweep_summary_checks_each_report(
    tmp_path: Path,
) -> None:
    report_paths = []
    for client_count in (1, 2):
        report_path = tmp_path / f"clients_{client_count:02d}" / "report.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(
            json.dumps(
                _report_payload(
                    client_count=client_count,
                    completed_rounds=1,
                    round_budget=1,
                )
            ),
            encoding="utf-8",
        )
        report_paths.append(report_path)

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "fl_ssl_client_count_sweep_summary.v1",
                "client_counts": [1, 2],
                "protocol": {"round_budget": 1},
                "runs": [
                    {"client_count": 1, "report_path": str(report_paths[0])},
                    {"client_count": 2, "report_path": str(report_paths[1])},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = verify_client_count_sweep_summary_path(
        summary_path,
        expected_client_counts=(1, 2),
        report_expectation=FederatedReportExpectation(
            expected_completed_rounds=1,
            expected_round_budget=1,
            expected_round_record_count=1,
            expected_round_update_count_matches_client_count=True,
            expected_ssl_algorithm="fixmatch",
            expected_ssl_method="fixmatch_usb_v1",
            expected_payload_adapter_kind="peft_classifier",
            expected_aggregation="fedavg",
            expected_delta_format="server_uploaded_artifact_ref",
        ),
    )

    assert result.passed


def test_verify_client_count_sweep_summary_returns_failure_for_missing_summary(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "missing.summary.json"

    result = verify_client_count_sweep_summary_path(
        summary_path,
        expected_client_counts=(1,),
        report_expectation=FederatedReportExpectation(),
    )

    assert not result.passed
    assert result.artifact == str(summary_path)
    assert result.errors
    assert result.errors[0].startswith("sweep summary file could not be read:")


def test_verify_client_count_sweep_summary_reports_nested_drift(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "clients_01" / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            _report_payload(
                client_count=1,
                completed_rounds=1,
                round_budget=1,
                ssl_algorithm="freematch",
            )
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "fl_ssl_client_count_sweep_summary.v1",
                "client_counts": [1],
                "protocol": {"round_budget": 1},
                "runs": [{"client_count": 1, "report_path": str(report_path)}],
            }
        ),
        encoding="utf-8",
    )

    result = verify_client_count_sweep_summary_path(
        summary_path,
        expected_client_counts=(1,),
        report_expectation=_expectation(client_count=None),
    )

    assert not result.passed
    assert (
        "client_count=1: objective.query_ssl.algorithm_name expected 'fixmatch', "
        "got 'freematch'." in result.errors
    )


def test_verify_artifact_manifest_checks_multiple_artifacts(
    tmp_path: Path,
    capsys,
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            _report_payload(
                client_count=1,
                completed_rounds=1,
                round_budget=1,
            )
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "fl_ssl_client_count_sweep_summary.v1",
                "client_counts": [1],
                "protocol": {"round_budget": 1},
                "runs": [{"client_count": 1, "report_path": str(report_path)}],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "expected_completed_rounds": 1,
                    "expected_round_budget": 1,
                    "expected_round_record_count": 1,
                    "expected_round_update_count_matches_client_count": True,
                    "expected_labeled_exposure_policy": "client_local_split",
                    "expected_run_control_budget_name": "main",
                    "expected_run_control_output_dir": "runs/fl_ssl",
                    "expected_ssl_algorithm": "fixmatch",
                    "expected_ssl_method": "fixmatch_usb_v1",
                    "expected_payload_adapter_kind": "peft_classifier",
                    "expected_aggregation": "fedavg",
                    "expected_delta_format": "server_uploaded_artifact_ref",
                },
                "artifacts": [
                    {
                        "name": "single_report",
                        "report": "report.json",
                        "expectation": {
                            "expected_client_count": 1,
                            "expected_seed": 42,
                            "expected_shard_policy_name": "dirichlet_label_skew",
                            "expected_shard_alpha": 0.3,
                            "expected_split_id_contains": "alpha0.3",
                        },
                    },
                    {
                        "name": "sweep_summary",
                        "client_count_sweep_summary": "summary.json",
                        "expected_client_counts": [1],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = verify_federated_report_artifacts_main(
        ["--manifest", str(manifest_path)]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "PASS single_report:" in output
    assert "PASS sweep_summary:" in output


def test_verify_artifact_manifest_applies_peft_snapshot_default(
    tmp_path: Path,
    capsys,
) -> None:
    report_path = _write_report_run_with_server_update_artifacts(
        tmp_path,
        payload=_report_payload(
            client_count=1,
            completed_rounds=1,
            round_budget=1,
            fallback_payload_adapter_kind="peft_classifier",
            objective_payload_scope="peft_classifier",
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "expected_payload_adapter_kind": "peft_classifier",
                    "expected_delta_format": "server_uploaded_artifact_ref",
                    "expected_shared_update_count_matches_round_updates": True,
                    "expect_server_owned_update_artifacts": True,
                    "expect_no_agent_local_update_refs": True,
                    "expected_aggregate_snapshot_update_family": "peft_text_encoder",
                },
                "artifacts": [
                    {
                        "name": "peft_report",
                        "report": str(report_path.relative_to(tmp_path)),
                        "expectation": {
                            "expected_completed_rounds": 1,
                            "expected_round_budget": 1,
                            "expected_client_count": 1,
                            "expected_round_record_count": 1,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = verify_federated_report_artifacts_main(
        ["--manifest", str(manifest_path)]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "PASS peft_report:" in output


def test_verify_artifact_cli_accepts_labeled_exposure_and_run_control_options(
    tmp_path: Path,
    capsys,
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            _report_payload(
                client_count=1,
                completed_rounds=1,
                round_budget=1,
                labeled_exposure_policy="shared_client_seed",
                run_control_budget_name="reduced",
                run_control_output_dir="runs/fl_ssl",
            )
        ),
        encoding="utf-8",
    )

    exit_code = verify_federated_report_artifacts_main(
        [
            "--report",
            str(report_path),
            "--expected-client-count",
            "1",
            "--expected-completed-rounds",
            "1",
            "--expected-round-budget",
            "1",
            "--expected-labeled-exposure-policy",
            "shared_client_seed",
            "--expected-run-control-budget-name",
            "reduced",
            "--expected-run-control-output-dir",
            "runs/fl_ssl",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"PASS {report_path}" in output


def test_verify_artifact_cli_accepts_federated_ssl_method_descriptor(
    tmp_path: Path,
    capsys,
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            _report_payload(
                client_count=1,
                completed_rounds=1,
                round_budget=1,
                fl_method_name="fedmatch",
                fl_method_descriptor_name="fedmatch",
                fl_method_execution_role="method_owned",
                federated_ssl_method="fedmatch",
                ssl_method_implementation_status=(
                    "partitioned_trainable_state_slice_v1"
                ),
                ssl_method_scenario="labels-at-client",
                ssl_method_local_budget_policy="iteration_capped",
                ssl_method_parameter_override_status="original",
            )
        ),
        encoding="utf-8",
    )

    exit_code = verify_federated_report_artifacts_main(
        [
            "--report",
            str(report_path),
            "--expected-fl-method-name",
            "fedmatch",
            "--expected-fl-method-descriptor-name",
            "fedmatch",
            "--expected-fl-method-execution-role",
            "method_owned",
            "--expected-federated-ssl-method",
            "fedmatch",
            "--expected-ssl-method-implementation-status",
            "partitioned_trainable_state_slice_v1",
            "--expected-ssl-method-scenario",
            "labels-at-client",
            "--expected-ssl-method-local-budget-policy",
            "iteration_capped",
            "--expected-ssl-method-parameter-override-status",
            "original",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"PASS {report_path}" in output


def test_verify_artifact_manifest_accepts_fedmatch_partitioned_expectations(
    tmp_path: Path,
    capsys,
) -> None:
    report_path = _write_report_run_with_server_update_artifacts(
        tmp_path,
        payload=_attach_artifact_communication_estimate(
            _report_payload(
                client_count=2,
                completed_rounds=2,
                round_budget=2,
                fl_method_name="fedmatch",
                fl_method_descriptor_name="fedmatch",
                fl_method_execution_role="method_owned",
                federated_ssl_method="fedmatch",
                ssl_method_implementation_status=(
                    "partitioned_trainable_state_slice_v1"
                ),
                ssl_method_scenario="labels-at-client",
                ssl_method_local_budget_policy="iteration_capped",
                ssl_method_parameter_override_status="original",
                labeled_exposure_policy="shared_client_seed",
                run_control_budget_name="reduced",
                server_update_policy="fedmatch_partitioned",
                update_partition_policy="partitioned",
                aggregation_weight_policy="uniform",
                peer_context_policy="fixed_probe_output_knn",
                local_ssl_policy="fedmatch_agreement",
            )
        ),
        partitioned_only=True,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "expected_seed": 42,
                    "expected_shard_policy_name": "dirichlet_label_skew",
                    "expected_shard_alpha": 0.3,
                    "expected_run_control_output_dir": "runs/fl_ssl",
                    "expected_delta_format": "server_uploaded_artifact_ref",
                    "expected_round_update_count_matches_client_count": True,
                    "expected_shared_update_count_matches_round_updates": True,
                    "expect_server_owned_update_artifacts": True,
                    "expect_no_agent_local_update_refs": True,
                    "expected_aggregate_snapshot_update_family": "peft_text_encoder",
                },
                "artifacts": [
                    {
                        "name": "fedmatch_reduced",
                        "report": "run/reports/fl_ssl_main_comparison.report.json",
                        "expectation": {
                            "expected_completed_rounds": 2,
                            "expected_round_budget": 2,
                            "expected_client_count": 2,
                            "expected_round_record_count": 2,
                            "expected_labeled_exposure_policy": "shared_client_seed",
                            "expected_run_control_budget_name": "reduced",
                            "expected_fl_method_name": "fedmatch",
                            "expected_fl_method_descriptor_name": ("fedmatch"),
                            "expected_fl_method_execution_role": "method_owned",
                            "expected_federated_ssl_method": ("fedmatch"),
                            "expected_ssl_method_implementation_status": (
                                "partitioned_trainable_state_slice_v1"
                            ),
                            "expected_ssl_method_scenario": "labels-at-client",
                            "expected_ssl_method_local_budget_policy": (
                                "iteration_capped"
                            ),
                            "expected_ssl_method_parameter_override_status": (
                                "original"
                            ),
                            "expected_server_update_policy": "fedmatch_partitioned",
                            "expected_update_partition_policy": "partitioned",
                            "expected_aggregation_weight_policy": "uniform",
                            "expected_peer_context_policy": "fixed_probe_output_knn",
                            "expected_local_ssl_policy": "fedmatch_agreement",
                            "expect_partitioned_update_artifact_refs": True,
                            "expected_communication_estimate_schema_version": (
                                communication_cost_estimates.COMMUNICATION_ESTIMATE_SCHEMA_VERSION
                            ),
                            "expect_partitioned_sparse_s2c_estimates": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = verify_federated_report_artifacts_main(
        ["--manifest", str(manifest_path)]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "PASS fedmatch_reduced:" in output
    assert str(report_path) in output


def test_verify_artifact_manifest_entry_expectation_overrides_defaults(
    tmp_path: Path,
    capsys,
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            _report_payload(
                client_count=1,
                completed_rounds=1,
                round_budget=1,
                ssl_algorithm="flexmatch",
                ssl_method="flexmatch_usb_v1",
            )
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "expected_ssl_algorithm": "fixmatch",
                    "expected_ssl_method": "fixmatch_usb_v1",
                },
                "artifacts": [
                    {
                        "name": "flexmatch_report",
                        "report": "report.json",
                        "expectation": {
                            "expected_ssl_algorithm": "flexmatch",
                            "expected_ssl_method": "flexmatch_usb_v1",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = verify_federated_report_artifacts_main(
        ["--manifest", str(manifest_path)]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "PASS flexmatch_report:" in output


def test_verify_artifact_manifest_keeps_repo_root_relative_artifact_paths(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "docs" / "operations" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    resolved = _optional_manifest_path(
        manifest_path,
        "runs/fl_ssl/example/reports/fl_ssl_main_comparison.report.json",
    )

    assert resolved == Path(
        "runs/fl_ssl/example/reports/fl_ssl_main_comparison.report.json"
    )


def test_verify_artifact_manifest_returns_failure_for_drift(
    tmp_path: Path,
    capsys,
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            _report_payload(
                client_count=1,
                completed_rounds=1,
                round_budget=1,
                ssl_algorithm="freematch",
            )
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "name": "drifted_report",
                        "report": "report.json",
                        "expectation": {
                            "expected_ssl_algorithm": "fixmatch",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = verify_federated_report_artifacts_main(
        ["--manifest", str(manifest_path)]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "FAIL drifted_report:" in output
    assert (
        "objective.query_ssl.algorithm_name expected 'fixmatch', got 'freematch'."
        in output
    )

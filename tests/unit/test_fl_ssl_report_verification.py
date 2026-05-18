"""FL SSL report artifact verifier tests."""

from __future__ import annotations

import json
from pathlib import Path

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
    ssl_algorithm: str = "fixmatch",
    ssl_method: str = "fixmatch_usb_v1",
    adapter_family: str = "lora_classifier",
    aggregation: str = "fedavg",
    delta_format: str = "server_uploaded_artifact_ref",
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
            "objective": {
                "query_ssl.algorithm_name": ssl_algorithm,
                "query_ssl.method_name": ssl_method,
                "lora_classifier.delta_format": delta_format,
            },
            "round_runtime": {
                "adapter_family_name": adapter_family,
                "aggregation_backend_name": aggregation,
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
        expected_adapter_family="lora_classifier",
        expected_aggregation="fedavg",
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


def _write_report_run_with_server_update_artifacts(
    tmp_path: Path,
    *,
    payload: dict[str, object],
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
            lora_ref = f"aggregation_artifact::{ref_prefix}/lora_delta"
            head_ref = f"aggregation_artifact::{ref_prefix}/classifier_head_delta"
            (artifact_root / ref_prefix).mkdir(parents=True)
            (artifact_root / f"{ref_prefix}/lora_delta.json").write_text(
                json.dumps({"lora_parameters": {"a": [1.0]}}),
                encoding="utf-8",
            )
            (artifact_root / f"{ref_prefix}/classifier_head_delta.json").write_text(
                json.dumps({"classifier_head_weights": {"normal": [1.0]}}),
                encoding="utf-8",
            )
            (update_dir / f"{update_id}.json").write_text(
                json.dumps(
                    {
                        "update_id": update_id,
                        "delta_format": "server_uploaded_artifact_ref",
                        "lora_delta_artifact_ref": lora_ref,
                        "classifier_head_delta_artifact_ref": head_ref,
                    }
                ),
                encoding="utf-8",
            )

    final_round = dict(payload["rounds"][-1])  # type: ignore[index,arg-type]
    snapshot_dir = (
        artifact_root / "lora_classifier" / str(final_round["model_revision"])
    )
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "lora_adapter.json").write_text(
        json.dumps({"lora_parameters": {"a": [1.0]}}),
        encoding="utf-8",
    )
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
            adapter_family="diagonal_scale",
        ),
        expectation=_expectation(),
    )

    assert not result.passed
    assert (
        "objective.query_ssl.algorithm_name expected 'fixmatch', got 'pseudolabel'."
        in result.errors
    )
    assert (
        "round_runtime.adapter_family_name expected 'lora_classifier', "
        "got 'diagonal_scale'." in result.errors
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
            expect_lora_classifier_aggregate_snapshot=True,
        ),
    )

    assert result.passed


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
    update_payload["lora_delta_artifact_ref"] = "agent-local://agent_01/lora_delta"
    update_path.write_text(json.dumps(update_payload), encoding="utf-8")

    result = verify_federated_simulation_report_path(
        report_path,
        FederatedReportExpectation(
            expected_delta_format="server_uploaded_artifact_ref",
            expected_shared_update_count_matches_round_updates=True,
            expect_server_owned_update_artifacts=True,
            expect_no_agent_local_update_refs=True,
            expect_lora_classifier_aggregate_snapshot=True,
        ),
    )

    assert not result.passed
    assert any(
        error.endswith("must not contain agent-local artifact refs.")
        for error in result.errors
    )
    assert any(
        ".lora_delta_artifact_ref must start with 'aggregation_artifact::'" in error
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
            expected_adapter_family="lora_classifier",
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
                    "expected_adapter_family": "lora_classifier",
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

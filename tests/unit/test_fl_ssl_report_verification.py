"""FL SSL report artifact verifier tests."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.fl_ssl.federated_simulation.io.report_verification import (
    FederatedReportExpectation,
    verify_client_count_sweep_summary_path,
    verify_federated_simulation_report_payload,
)


def _report_payload(
    *,
    client_count: int,
    completed_rounds: int,
    round_budget: int,
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
) -> dict[str, object]:
    return {
        "schema_version": "fl_ssl_main_comparison.v1",
        "protocol": {
            "client_count": client_count,
            "completed_rounds": completed_rounds,
            "round_budget": round_budget,
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
        report_expectation=_expectation(client_count=None),
    )

    assert result.passed


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

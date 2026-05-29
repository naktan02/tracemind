"""FL SSL artifact communication cost estimate 검증."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.fl_ssl.federated_simulation.io import (
    communication_cost_estimates,
)


def test_build_artifact_communication_estimate_estimates_c2s_and_s2c(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "fl_ssl" / "manual" / "method" / "split" / "r"
    _write_json(
        run_dir
        / "main_server"
        / "shared_adapter_states"
        / "versions"
        / "sim_rev_0000.json",
        {
            "peft_adapter_artifact_ref": (
                "server-aggregate://peft_classifier/sim_rev_0000/peft_adapter"
            ),
            "classifier_head_artifact_ref": (
                "server-aggregate://peft_classifier/sim_rev_0000/classifier_head"
            ),
        },
    )
    _write_json(
        run_dir
        / "main_server"
        / "shared_adapter_states"
        / "versions"
        / "sim_rev_0001.json",
        {
            "peft_adapter_artifact_ref": (
                "server-aggregate://peft_classifier/sim_rev_0001/peft_adapter"
            ),
            "classifier_head_artifact_ref": (
                "server-aggregate://peft_classifier/sim_rev_0001/classifier_head"
            ),
        },
    )
    _write_bytes(
        run_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "peft_classifier"
        / "sim_rev_0000"
        / "peft_adapter.json",
        100,
    )
    _write_bytes(
        run_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "peft_classifier"
        / "sim_rev_0000"
        / "classifier_head.json",
        20,
    )
    _write_json(
        run_dir / "main_server" / "rounds" / "records" / "round_0001.json",
        {
            "active_manifest": {"model_revision": "sim_rev_0000"},
            "training_task": {"round_id": "round_0001"},
        },
    )
    _write_json(
        run_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "peft_classifier"
        / "sim_rev_0001"
        / "peft_adapter.json",
        {
            "partitioned_peft_parameters": {
                "sigma": {"encoder_lora.weight": [1.0, 0.0]},
                "psi": {"encoder_lora.weight": [0.0, 2.0]},
            }
        },
    )
    estimate = communication_cost_estimates.build_artifact_communication_estimate(
        run_dir=run_dir,
        rounds=[
            {
                "round_id": "round_0001",
                "round_index": 1,
                "selected_client_count": 2,
                "total_payload_bytes": 7,
                "clients": [
                    {
                        "client_payload_bytes": 3,
                        "client_artifact_bytes": 10,
                    },
                    {
                        "client_payload_bytes": 4,
                        "client_artifact_bytes": 11,
                    },
                ],
            }
        ],
    )

    assert estimate["schema_version"] == (
        communication_cost_estimates.COMMUNICATION_ESTIMATE_SCHEMA_VERSION
    )
    assert estimate["c2s_payload_bytes"] == 7
    assert estimate["c2s_artifact_bytes"] == 21
    assert estimate["c2s_total_bytes"] == 28
    assert estimate["s2c_global_state_bytes_estimated"] == 240
    assert estimate["s2c_total_bytes_estimated"] > 240


def test_build_artifact_communication_estimate_estimates_partitioned_sparse_s2c(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "fl_ssl" / "fedmatch" / "split" / "r"
    _write_json(
        run_dir
        / "main_server"
        / "shared_adapter_states"
        / "versions"
        / "sim_rev_0000.json",
        {
            "peft_adapter_artifact_ref": (
                "server-aggregate://peft_classifier/sim_rev_0000/peft_adapter"
            ),
            "classifier_head_artifact_ref": (
                "server-aggregate://peft_classifier/sim_rev_0000/classifier_head"
            ),
        },
    )
    _write_json(
        run_dir
        / "main_server"
        / "shared_adapter_states"
        / "versions"
        / "sim_rev_0001.json",
        {
            "peft_adapter_artifact_ref": (
                "server-aggregate://peft_classifier/sim_rev_0001/peft_adapter"
            ),
            "classifier_head_artifact_ref": (
                "server-aggregate://peft_classifier/sim_rev_0001/classifier_head"
            ),
        },
    )
    _write_json(
        run_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "peft_classifier"
        / "sim_rev_0001"
        / "peft_adapter.json",
        {
            "partitioned_peft_parameters": {
                "sigma": {"encoder_lora.weight": [1.0, 0.0]},
                "psi": {"encoder_lora.weight": [0.0, 2.0]},
            }
        },
    )
    _write_json(
        run_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "peft_classifier"
        / "sim_rev_0001"
        / "classifier_head.json",
        {
            "partitioned_classifier_head_weights": {
                "sigma": {"anxiety": [3.0, 0.0]},
            },
            "partitioned_classifier_head_biases": {
                "psi": {"anxiety": -1.0},
            },
        },
    )
    _write_json(
        run_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "peft_classifier"
        / "sim_rev_0000"
        / "peft_adapter.json",
        {
            "partitioned_peft_parameters": {
                "sigma": {"encoder_lora.weight": [1.0, 0.0]},
                "psi": {"encoder_lora.weight": [0.0, 2.0]},
            }
        },
    )
    _write_json(
        run_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "peft_classifier"
        / "sim_rev_0000"
        / "classifier_head.json",
        {
            "partitioned_classifier_head_weights": {
                "sigma": {"anxiety": [3.0, 0.0]},
            },
            "partitioned_classifier_head_biases": {
                "psi": {"anxiety": -1.0},
            },
        },
    )
    estimate = communication_cost_estimates.build_artifact_communication_estimate(
        run_dir=run_dir,
        rounds=[
            {
                "round_id": "round_0001",
                "round_index": 1,
                "selected_client_count": 3,
                "clients": [],
            },
            {
                "round_id": "round_0002",
                "round_index": 2,
                "selected_client_count": 3,
                "clients": [],
            },
        ],
    )

    assert estimate["s2c_partitioned_sparse_transport_bytes_estimated"] == 96
    assert estimate["s2c_total_bytes_estimated"] == (
        estimate["s2c_global_state_bytes_estimated"] + 96
    )
    assert estimate["bidirectional_total_bytes_estimated"] == (
        estimate["c2s_total_bytes"] + estimate["s2c_total_bytes_estimated"]
    )
    assert (
        estimate["per_round"][0]["s2c_partitioned_sparse_transport_bytes_estimated"]
        == 0
    )
    assert (
        estimate["per_round"][1]["s2c_partitioned_sparse_transport_bytes_estimated"]
        == 96
    )
    assert estimate["per_round"][1]["s2c_total_bytes_estimated"] == (
        estimate["per_round"][1]["s2c_global_state_bytes_estimated"] + 96
    )


def test_attach_artifact_communication_estimate_updates_report_cost(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    communication_cost: dict[str, object] = {"value": 1}

    communication_cost_estimates.attach_artifact_communication_estimate(
        communication_cost=communication_cost,
        run_dir=run_dir,
        rounds=[
            {
                "round_id": "round_0001",
                "round_index": 1,
                "selected_client_count": 1,
                "clients": [],
            }
        ],
    )

    estimate = communication_cost["artifact_byte_estimates"]
    assert isinstance(estimate, dict)
    assert estimate["schema_version"] == (
        communication_cost_estimates.COMMUNICATION_ESTIMATE_SCHEMA_VERSION
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_bytes(path: Path, byte_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * byte_count)

"""FL SSL posthoc communication cost backfill 검증."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.fl_ssl.backfill_communication_costs import (
    POSTHOC_SCHEMA_VERSION,
    build_posthoc_communication_cost,
    write_posthoc_communication_cost,
)


def test_build_posthoc_communication_cost_estimates_c2s_and_s2c(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "fl_ssl" / "manual" / "method" / "split" / "r"
    report_path = run_dir / "reports" / "fl_ssl_main_comparison.report.json"
    _write_json(
        run_dir
        / "main_server"
        / "shared_adapter_states"
        / "versions"
        / "sim_rev_0000.json",
        {
            "lora_adapter_artifact_ref": (
                "server-aggregate://lora_classifier/sim_rev_0000/lora_adapter"
            ),
            "classifier_head_artifact_ref": (
                "server-aggregate://lora_classifier/sim_rev_0000/classifier_head"
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
            "lora_adapter_artifact_ref": (
                "server-aggregate://lora_classifier/sim_rev_0001/lora_adapter"
            ),
            "classifier_head_artifact_ref": (
                "server-aggregate://lora_classifier/sim_rev_0001/classifier_head"
            ),
        },
    )
    _write_bytes(
        run_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "lora_classifier"
        / "sim_rev_0000"
        / "lora_adapter.json",
        100,
    )
    _write_bytes(
        run_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "lora_classifier"
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
        / "lora_classifier"
        / "sim_rev_0001"
        / "lora_adapter.json",
        {
            "partitioned_lora_parameters": {
                "sigma": {"encoder_lora.weight": [1.0, 0.0]},
                "psi": {"encoder_lora.weight": [0.0, 2.0]},
            }
        },
    )
    payload = {
        "track": "fl_ssl_main_comparison",
        "metrics": {"secondary": {"communication_cost": {"value": 2}}},
        "diagnostics": {"communication_cost": {"value": 2}},
        "rounds": [
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
    }

    posthoc = build_posthoc_communication_cost(
        report_path=report_path,
        payload=payload,
    )

    assert posthoc["schema_version"] == POSTHOC_SCHEMA_VERSION
    assert posthoc["c2s_payload_bytes"] == 7
    assert posthoc["c2s_artifact_bytes"] == 21
    assert posthoc["c2s_total_bytes"] == 28
    assert posthoc["s2c_global_state_bytes_estimated"] == 240
    assert posthoc["s2c_total_bytes_estimated"] > 240


def test_build_posthoc_communication_cost_estimates_partitioned_sparse_s2c(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "fl_ssl" / "fedmatch" / "split" / "r"
    report_path = run_dir / "reports" / "fl_ssl_main_comparison.report.json"
    _write_json(
        run_dir
        / "main_server"
        / "shared_adapter_states"
        / "versions"
        / "sim_rev_0000.json",
        {
            "lora_adapter_artifact_ref": (
                "server-aggregate://lora_classifier/sim_rev_0000/lora_adapter"
            ),
            "classifier_head_artifact_ref": (
                "server-aggregate://lora_classifier/sim_rev_0000/classifier_head"
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
            "lora_adapter_artifact_ref": (
                "server-aggregate://lora_classifier/sim_rev_0001/lora_adapter"
            ),
            "classifier_head_artifact_ref": (
                "server-aggregate://lora_classifier/sim_rev_0001/classifier_head"
            ),
        },
    )
    _write_json(
        run_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "lora_classifier"
        / "sim_rev_0001"
        / "lora_adapter.json",
        {
            "partitioned_lora_parameters": {
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
        / "lora_classifier"
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
        / "lora_classifier"
        / "sim_rev_0000"
        / "lora_adapter.json",
        {
            "partitioned_lora_parameters": {
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
        / "lora_classifier"
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
    payload = {
        "rounds": [
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
    }

    posthoc = build_posthoc_communication_cost(
        report_path=report_path,
        payload=payload,
    )

    assert posthoc["s2c_partitioned_sparse_transport_bytes_estimated"] == 96
    assert posthoc["s2c_total_bytes_estimated"] == (
        posthoc["s2c_global_state_bytes_estimated"] + 96
    )
    assert posthoc["bidirectional_total_bytes_estimated"] == (
        posthoc["c2s_total_bytes"] + posthoc["s2c_total_bytes_estimated"]
    )
    assert (
        posthoc["per_round"][0]["s2c_partitioned_sparse_transport_bytes_estimated"] == 0
    )
    assert (
        posthoc["per_round"][1]["s2c_partitioned_sparse_transport_bytes_estimated"]
        == 96
    )
    assert posthoc["per_round"][1]["s2c_total_bytes_estimated"] == (
        posthoc["per_round"][1]["s2c_global_state_bytes_estimated"] + 96
    )


def test_write_posthoc_communication_cost_merges_report_and_sidecar(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "run" / "reports" / "fl_ssl_main_comparison.report.json"
    report_path.parent.mkdir(parents=True)
    payload = {
        "metrics": {"secondary": {"communication_cost": {"value": 1}}},
        "diagnostics": {"communication_cost": {"value": 1}},
    }
    posthoc = {
        "schema_version": POSTHOC_SCHEMA_VERSION,
        "c2s_total_bytes": 10,
    }

    write_posthoc_communication_cost(
        report_path=report_path,
        payload=payload,
        posthoc=posthoc,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    sidecar = json.loads(
        (report_path.parent / "fl_ssl_posthoc_communication_cost.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar == posthoc
    assert report["diagnostics"]["communication_cost"]["posthoc_byte_estimates"] == (
        posthoc
    )
    assert (
        report["metrics"]["secondary"]["communication_cost"]["posthoc_byte_estimates"]
        == posthoc
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_bytes(path: Path, byte_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * byte_count)

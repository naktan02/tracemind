"""Experiment result index 검증."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.workflows.result_index.dashboard_export import write_dashboard_bundle
from scripts.workflows.result_index.ingest import ingest_reports
from scripts.workflows.result_index.report_loader import (
    discover_report_paths,
    load_result_index_records,
)
from scripts.workflows.result_index.sqlite_store import (
    initialize_database,
    write_result_index_records,
)

PEFT_ADAPTER_PARAMETERS_JSON = (
    '{"alpha":16,"bias":"none","dropout":0.1,"rank":8,'
    '"target_modules":"all-linear","use_rslora":false}'
)


def test_load_result_index_records_normalizes_report_shape(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = payload["manifest"]

    records = load_result_index_records(report_path)

    assert "ssl_input_mode" not in manifest
    assert "pseudo_label_algorithm" not in manifest
    assert "teacher_provider" not in manifest
    assert records.run.run_id == "peft_fixmatch_2026_05_13_143419"
    assert records.run.track == "central_peft_ssl"
    assert records.run.method_family == "peft_classifier"
    assert records.run.method_name == "fixmatch_usb_v1"
    assert records.run.algorithm_name == "fixmatch"
    assert records.run.labeled_dataset_name == "ourafla_reddit"
    assert records.run.unlabeled_dataset_name == "szegeelim_general4"
    assert records.run.validation_dataset_name is None
    assert records.run.test_dataset_name == "ourafla_reddit"
    assert records.run.run_control_budget_name == "main"
    assert records.run.run_control_output_dir == "runs"
    assert records.run.peft_adapter_name == "lora"
    assert records.run.peft_adapter_rank == 8
    assert records.run.peft_adapter_alpha == 16
    assert records.run.peft_adapter_dropout == 0.1
    assert records.run.peft_adapter_bias == "none"
    assert records.run.peft_adapter_target_modules == "all-linear"
    assert records.run.peft_adapter_parameters_json == PEFT_ADAPTER_PARAMETERS_JSON
    assert records.eval_metrics[0].macro_f1 == 0.78
    assert records.per_class_metrics[0].category == "anxiety"
    assert records.confusion_matrix_cells[0].actual_category == "anxiety"
    assert records.epoch_metrics[0].step == 2000
    assert records.epoch_metrics[0].selection_macro_f1 == 0.74
    normal_epoch_metric = next(
        metric
        for metric in records.epoch_per_class_metrics
        if metric.category == "normal"
    )
    assert normal_epoch_metric.f1 == 0.7
    assert records.artifacts[0].artifact_kind == "projection_manifest"
    assert records.artifacts[1].artifact_kind == "projection_points_jsonl"
    assert records.artifacts[2].artifact_kind == "projection_png"


def test_load_result_index_records_keeps_peft_track(
    tmp_path: Path,
) -> None:
    report_path = _write_peft_report(tmp_path)

    records = load_result_index_records(report_path)

    assert records.run.track == "central_peft_ssl"
    assert records.run.method_family == "peft_classifier"
    assert records.run.method_name == "fixmatch_usb_v1"


def test_load_result_index_records_keeps_legacy_peft_entrypoint_track(
    tmp_path: Path,
) -> None:
    report_path = _write_legacy_entrypoint_report(tmp_path)

    records = load_result_index_records(report_path)

    assert records.run.track == "central_peft_ssl"
    assert records.run.method_family == "peft_classifier"


def test_load_result_index_records_ignores_removed_backbone_lora_key(
    tmp_path: Path,
) -> None:
    report_path = _write_report(tmp_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    backbone = payload["manifest"]["backbone"]
    backbone.pop("peft_adapter_config")
    backbone["lora"] = {
        "adapter_name": "lora",
        "rank": 8,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": "all-linear",
        "use_rslora": False,
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    records = load_result_index_records(report_path)

    assert records.run.peft_adapter_name is None
    assert records.run.peft_adapter_rank is None


def test_write_result_index_records_and_export_dashboard_json(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path)
    db_path = tmp_path / "experiment_results.sqlite"
    dashboard_path = tmp_path / "experiment_dashboard.json"
    records = load_result_index_records(report_path)

    write_result_index_records(db_path=db_path, records=[records])
    bundle = write_dashboard_bundle(db_path=db_path, output_path=dashboard_path)

    assert dashboard_path.exists()
    assert bundle["filters"]["methods"] == ["fixmatch_usb_v1"]
    assert bundle["filters"]["run_control_budget_names"] == ["main"]
    assert bundle["filters"]["run_control_output_dirs"] == ["runs"]
    assert bundle["runs"][0]["selection_slug"] == (
        "labeled-ourafla_reddit_unlabeled-szegeelim_general4_test-ourafla_reddit"
    )
    assert bundle["projection_images"][0]["image_src"].startswith(
        "data/artifacts/peft_fixmatch_2026_05_13_143419/"
    )
    assert "data/datasets" not in json.dumps(bundle)

    with sqlite3.connect(db_path) as connection:
        per_class_count = connection.execute(
            "select count(*) from per_class_metrics"
        ).fetchone()[0]
        confusion_count = connection.execute(
            "select count(*) from confusion_matrix_cells"
        ).fetchone()[0]
        artifact_count = connection.execute(
            "select count(*) from artifacts"
        ).fetchone()[0]
    assert per_class_count == 2
    assert confusion_count == 4
    assert artifact_count == 3


def test_result_index_schema_migration_adds_run_control_columns(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "experiment_results.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            create table experiment_runs (
                run_id text primary key,
                track text not null,
                method_family text not null,
                method_name text not null
            )
            """
        )

    initialize_database(db_path)

    with sqlite3.connect(db_path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute(
                "pragma table_info(experiment_runs)"
            ).fetchall()
        }
    assert "run_control_budget_name" in columns
    assert "run_control_output_dir" in columns
    assert "labeled_row_exposure_count" in columns
    assert "unique_labeled_row_count" in columns
    assert "peft_adapter_rank" in columns
    assert "peft_adapter_parameters_json" in columns
    assert "update_family_name" in columns


def test_load_result_index_records_normalizes_fl_ssl_report_shape(
    tmp_path: Path,
) -> None:
    report_path = _write_fl_ssl_report(tmp_path)

    records = load_result_index_records(report_path)

    assert records.run.run_id == (
        "fixmatch_peft_adapter_alpha03_10c_50round_20260518__20260517T150549Z"
    )
    assert records.run.track == "fl_ssl_main_comparison"
    assert records.run.method_family == "manual_baselines"
    assert records.run.method_name == "fixmatch_usb_v1"
    assert records.run.algorithm_name == "fixmatch"
    assert records.run.selection_slug == (
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_"
        "test-ourafla_reddit_"
        "dirichlet_label_skew_clients10_seed42"
    )
    assert records.run.labeled_dataset_name == "ourafla_reddit"
    assert records.run.unlabeled_dataset_name == "ourafla_reddit"
    assert records.run.seed == 42
    assert records.run.client_count == 10
    assert records.run.round_budget == 50
    assert records.run.completed_rounds == 50
    assert records.run.run_control_budget_name == "main"
    assert records.run.run_control_output_dir == "runs/fl_ssl"
    assert records.run.total_row_exposure_count == 40960
    assert records.run.labeled_row_exposure_count == 405
    assert records.run.unlabeled_row_exposure_count == 40555
    assert records.run.unique_total_row_count == 40596
    assert records.run.unique_labeled_row_count == 41
    assert records.run.unique_unlabeled_row_count == 40555
    assert records.run.shard_policy_name == "dirichlet_label_skew"
    assert records.run.shard_alpha == 0.3
    assert records.run.payload_adapter_kind == "peft_classifier"
    assert records.run.update_family_name == "peft_text_encoder"
    assert records.run.aggregation_backend_name == "fedavg"
    assert records.run.fl_composition_mode == "manual"
    assert records.run.fl_execution_role == "manual_baseline"
    assert records.run.fl_descriptor_name is None
    assert records.run.update_delta_format == "server_uploaded_artifact_ref"
    assert records.run.peft_adapter_name == "lora"
    assert records.run.peft_adapter_rank == 8
    assert records.run.peft_adapter_alpha == 16
    assert records.run.peft_adapter_dropout == 0.1
    assert records.run.peft_adapter_target_modules == "all-linear"
    assert records.run.peft_adapter_parameters_json == PEFT_ADAPTER_PARAMETERS_JSON
    assert records.run.embedding_backend == "transformers_mxbai"
    assert records.run.embedding_device == "cuda"
    assert records.eval_metrics[1].eval_set == "final_validation"
    assert records.eval_metrics[1].macro_f1 == 0.78
    assert records.per_class_metrics[0].category == "anxiety"
    assert records.confusion_matrix_cells[0].actual_category == "anxiety"
    assert records.artifacts[0].artifact_kind == "fl_ssl_report"
    assert records.artifacts[1].artifact_kind == "fl_client_split_manifest"


def test_load_result_index_records_reads_peft_classifier_objective(
    tmp_path: Path,
) -> None:
    report_path = _write_peft_fl_ssl_report(tmp_path)

    records = load_result_index_records(report_path)

    assert records.run.method_family == "manual_baselines"
    assert records.run.payload_adapter_kind == "peft_classifier"
    assert records.run.update_family_name == "peft_text_encoder"
    assert records.run.peft_adapter_name == "lora"
    assert records.run.peft_adapter_rank == 8
    assert records.run.peft_adapter_alpha == 16
    assert records.run.peft_adapter_dropout == 0.1
    assert records.run.peft_adapter_bias == "none"
    assert records.run.peft_adapter_target_modules == "all-linear"
    assert records.run.peft_adapter_parameters_json == PEFT_ADAPTER_PARAMETERS_JSON
    assert records.run.update_delta_format == "server_uploaded_artifact_ref"


def test_fl_ssl_result_index_reads_payload_adapter_kind(
    tmp_path: Path,
) -> None:
    report_path = _write_peft_fl_ssl_report(tmp_path)
    payload = _sample_fl_ssl_report()
    protocol = payload["protocol"]
    protocol["round_runtime"] = {
        "payload_adapter_kind": "peft_classifier",
        "update_family_name": "peft_text_encoder",
        "aggregation_backend_name": "fedavg",
    }
    protocol["objective"] = {
        "query_ssl.method_name": "fixmatch_usb_v1",
        "query_ssl.algorithm_name": "fixmatch",
        "peft_classifier.peft_adapter_name": "lora",
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    records = load_result_index_records(report_path)

    assert records.run.payload_adapter_kind == "peft_classifier"


def test_result_index_discovers_fl_ssl_report_artifacts(tmp_path: Path) -> None:
    central_report = _write_report(tmp_path)
    fl_report = _write_fl_ssl_report(tmp_path)

    assert discover_report_paths(tmp_path / "runs") == [fl_report, central_report]


def test_result_index_excludes_smoke_reports_from_default_runs_ingest(
    tmp_path: Path,
) -> None:
    central_report = _write_report(tmp_path)
    smoke_report = _write_smoke_report(tmp_path)
    metadata_smoke_report = _write_metadata_smoke_report(tmp_path)

    assert discover_report_paths(tmp_path / "runs") == [central_report]
    assert discover_report_paths(tmp_path / "runs" / "_smoke") == [smoke_report]
    assert discover_report_paths(
        tmp_path / "runs" / "_smoke" / "run_peft_ssl_control"
    ) == [smoke_report]
    assert metadata_smoke_report not in discover_report_paths(tmp_path / "runs")


def test_result_index_default_ingest_keeps_smoke_out_of_dashboard(
    tmp_path: Path,
) -> None:
    _write_report(tmp_path)
    _write_smoke_report(tmp_path)
    _write_metadata_smoke_report(tmp_path)
    db_path = tmp_path / "experiment_results.sqlite"
    dashboard_path = tmp_path / "experiment_dashboard.json"

    indexed_count = ingest_reports(
        runs_root=tmp_path / "runs",
        db_path=db_path,
        reset=True,
    )
    bundle = write_dashboard_bundle(db_path=db_path, output_path=dashboard_path)

    assert indexed_count == 1
    assert [run["run_id"] for run in bundle["runs"]] == [
        "peft_fixmatch_2026_05_13_143419"
    ]
    assert bundle["filters"]["run_control_budget_names"] == ["main"]
    assert bundle["filters"]["run_control_output_dirs"] == ["runs"]


def test_result_index_prefers_canonical_fl_ssl_hardlink_path(
    tmp_path: Path,
) -> None:
    old_report = _write_fl_ssl_report(tmp_path)
    canonical_report = (
        tmp_path
        / "runs"
        / "fl_ssl"
        / "manual_baselines"
        / "fixmatch_usb_v1__peft_text_encoder_lora__fedavg"
        / "alpha03_seed42"
        / "clients10_rounds50"
        / "20260517T150549Z"
        / "reports"
        / "fl_ssl_main_comparison.report.json"
    )
    canonical_report.parent.mkdir(parents=True, exist_ok=True)
    canonical_report.hardlink_to(old_report)

    assert discover_report_paths(tmp_path / "runs") == [canonical_report]


def test_load_result_index_records_keeps_client_count_sweep_slug(
    tmp_path: Path,
) -> None:
    report_path = _write_fl_ssl_sweep_report(tmp_path, client_slug="clients_03")

    records = load_result_index_records(report_path)

    assert records.run.run_id == (
        "fixmatch_peft_adapter_alpha03_1round_20260518__20260517T193320Z__clients_03"
    )


def test_load_result_index_records_keeps_new_fl_ssl_layout_parts(
    tmp_path: Path,
) -> None:
    report_path = _write_new_layout_fl_ssl_report(tmp_path)

    records = load_result_index_records(report_path)

    assert records.run.run_id == (
        "manual_baselines__fixmatch_usb_v1__peft_text_encoder_lora__fedavg__"
        "alpha03_seed42__clients10_rounds50__20260518T010203Z"
    )


def test_write_result_index_records_exports_fl_ssl_dashboard_filters(
    tmp_path: Path,
) -> None:
    report_path = _write_fl_ssl_report(tmp_path)
    db_path = tmp_path / "experiment_results.sqlite"
    dashboard_path = tmp_path / "experiment_dashboard.json"
    records = load_result_index_records(report_path)

    write_result_index_records(db_path=db_path, records=[records])
    bundle = write_dashboard_bundle(db_path=db_path, output_path=dashboard_path)

    assert bundle["filters"]["tracks"] == ["fl_ssl_main_comparison"]
    assert bundle["filters"]["methods"] == ["fixmatch_usb_v1"]
    assert bundle["filters"]["algorithms"] == ["fixmatch"]
    assert bundle["filters"]["method_families"] == ["manual_baselines"]
    assert bundle["filters"]["fl_composition_modes"] == ["manual"]
    assert bundle["filters"]["fl_execution_roles"] == ["manual_baseline"]
    assert bundle["filters"]["fl_descriptors"] == []
    assert bundle["filters"]["run_control_budget_names"] == ["main"]
    assert bundle["filters"]["run_control_output_dirs"] == ["runs/fl_ssl"]
    assert bundle["filters"]["client_counts"] == [10]
    assert bundle["filters"]["round_budgets"] == [50]
    assert bundle["filters"]["shard_alphas"] == [0.3]
    assert bundle["filters"]["payload_adapter_kinds"] == ["peft_classifier"]
    assert bundle["filters"]["peft_adapter_names"] == ["lora"]
    assert bundle["filters"]["peft_adapter_ranks"] == [8]
    assert bundle["filters"]["peft_adapter_alphas"] == [16]
    assert bundle["filters"]["peft_adapter_parameter_snapshots"] == [
        PEFT_ADAPTER_PARAMETERS_JSON
    ]
    assert bundle["filters"]["aggregation_backends"] == ["fedavg"]
    assert bundle["filters"]["update_delta_formats"] == ["server_uploaded_artifact_ref"]
    assert bundle["filters"]["embedding_backends"] == ["transformers_mxbai"]
    assert bundle["filters"]["embedding_model_ids"] == [
        "mixedbread-ai/mxbai-embed-large-v1"
    ]
    assert bundle["filters"]["embedding_devices"] == ["cuda"]
    assert bundle["filters"]["local_trainer_devices"] == ["cuda"]
    assert bundle["fl_ssl_runs"][0]["macro_f1"] == 0.78
    assert bundle["fl_ssl_runs"][0]["worst_client_macro_f1"] == 0.41
    assert bundle["fl_ssl_runs"][0]["expected_calibration_error"] == 0.12
    assert bundle["fl_ssl_runs"][0]["labeled_row_exposure_count"] == 405
    assert bundle["fl_ssl_runs"][0]["unique_labeled_row_count"] == 41
    assert bundle["fl_ssl_runs"][0]["communication_cost"]["value"] == 500
    assert bundle["fl_ssl_runs"][0]["per_client_macro_f1_variance"] == 0.02
    assert bundle["fl_ssl_runs"][0]["macro_f1_std"] == 0.1
    assert [row["round_index"] for row in bundle["fl_ssl_rounds"]] == [0, 1, 2]
    assert bundle["fl_ssl_rounds"][1]["round_id"] == "round_0001"
    assert bundle["fl_ssl_rounds"][1]["update_count"] == 10
    assert bundle["fl_ssl_rounds"][1]["accepted_ratio"] == pytest.approx(7 / 12)
    assert bundle["fl_ssl_rounds"][1]["round_update_delta_l2_mean"] == pytest.approx(
        1.1
    )
    assert bundle["fl_ssl_rounds"][1]["round_update_delta_l2_max"] == pytest.approx(1.1)
    assert bundle["fl_ssl_rounds"][2]["macro_f1_delta_from_initial"] == 0.08
    assert len(bundle["fl_ssl_client_rounds"]) == 2
    assert bundle["fl_ssl_client_rounds"][0]["client_id"] == "agent_01"
    assert bundle["fl_ssl_client_rounds"][0]["accepted_count"] == 7
    assert bundle["fl_ssl_client_rounds"][1]["delta_l2_norm"] == 1.25
    assert bundle["fl_ssl_client_rounds"][1]["per_client_delta_l2_norm"] == 1.25
    assert bundle["fl_ssl_client_validations"][0]["client_id"] == "agent_01"
    assert bundle["fl_ssl_client_validations"][0]["client_validation_macro_f1"] == 0.41
    assert bundle["fl_ssl_client_splits"][0]["source"] == "client_validation"
    assert bundle["fl_ssl_client_splits"][0]["labeled_count"] == 8
    assert bundle["fl_ssl_client_splits"][0]["label_distribution"] == {
        "anxiety": 8,
        "normal": 12,
    }
    assert bundle["projection_images"][0]["run_id"] == (
        "fixmatch_peft_adapter_alpha03_10c_50round_20260518__20260517T150549Z"
    )
    assert bundle["projection_images"][0]["eval_set"] == "validation"


def test_fl_ssl_dashboard_filters_deduplicate_peft_classifier_runs(
    tmp_path: Path,
) -> None:
    fl_report_path = _write_fl_ssl_report(tmp_path)
    peft_report_path = _write_peft_fl_ssl_report(tmp_path)
    db_path = tmp_path / "experiment_results.sqlite"
    dashboard_path = tmp_path / "experiment_dashboard.json"
    records = [
        load_result_index_records(fl_report_path),
        load_result_index_records(peft_report_path),
    ]

    write_result_index_records(db_path=db_path, records=records)
    bundle = write_dashboard_bundle(db_path=db_path, output_path=dashboard_path)

    assert bundle["filters"]["payload_adapter_kinds"] == ["peft_classifier"]
    assert bundle["filters"]["update_families"] == [
        "peft_text_encoder",
    ]
    assert bundle["filters"]["aggregation_backends"] == ["fedavg"]
    assert bundle["filters"]["peft_adapter_names"] == ["lora"]
    assert {
        row["payload_adapter_kind"]
        for row in bundle["fl_ssl_runs"]
        if row["track"] == "fl_ssl_main_comparison"
    } == {"peft_classifier"}


def _write_report(tmp_path: Path) -> Path:
    report_path = (
        tmp_path
        / "runs"
        / "run_peft_ssl_control"
        / "consistency"
        / ("labeled-ourafla_reddit_unlabeled-szegeelim_general4_test-ourafla_reddit")
        / "fixmatch_usb_v1"
        / "peft_fixmatch_2026_05_13_143419"
        / "reports"
        / "report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    projection_dir = report_path.parent.parent / "projections"
    projection_dir.mkdir(parents=True, exist_ok=True)
    (projection_dir / "projection_manifest.json").write_text("{}\n", encoding="utf-8")
    (projection_dir / "validation.projection.jsonl").write_text("", encoding="utf-8")
    (projection_dir / "validation.projection.png").write_bytes(b"png")
    report_path.write_text(
        json.dumps(_sample_report(projection_dir), indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_peft_report(tmp_path: Path) -> Path:
    report_path = (
        tmp_path
        / "runs"
        / "run_peft_ssl_control"
        / "consistency"
        / ("labeled-ourafla_reddit_unlabeled-szegeelim_general4_test-ourafla_reddit")
        / "fixmatch_usb_v1"
        / "peft_fixmatch_2026_05_13_143419"
        / "reports"
        / "report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    projection_dir = report_path.parent.parent / "projections"
    projection_dir.mkdir(parents=True, exist_ok=True)
    payload = _sample_report(projection_dir)
    payload["schema_version"] = "central_peft_classifier_eval.v1"
    report_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_fl_ssl_report(tmp_path: Path) -> Path:
    report_path = (
        tmp_path
        / "runs"
        / "federated_simulation"
        / "fixmatch_peft_adapter_alpha03_10c_50round_20260518"
        / "20260517T150549Z"
        / "reports"
        / "fl_ssl_main_comparison.report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    projection_dir = report_path.parent.parent / "projections"
    projection_dir.mkdir(parents=True, exist_ok=True)
    (projection_dir / "projection_manifest.json").write_text("{}\n", encoding="utf-8")
    (projection_dir / "validation.projection.jsonl").write_text("", encoding="utf-8")
    (projection_dir / "validation.projection.png").write_bytes(b"png")
    report_path.write_text(
        json.dumps(_sample_fl_ssl_report(projection_dir), indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_peft_fl_ssl_report(tmp_path: Path) -> Path:
    report_path = (
        tmp_path
        / "runs"
        / "fl_ssl"
        / "manual_baselines"
        / "fixmatch_usb_v1__peft_text_encoder_lora__fedavg"
        / "alpha03_seed42"
        / "clients10_rounds50"
        / "20260527T120000Z"
        / "reports"
        / "fl_ssl_main_comparison.report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _sample_fl_ssl_report()
    protocol = payload["protocol"]
    protocol["round_runtime"] = {
        "payload_adapter_kind": "peft_classifier",
        "update_family_name": "peft_text_encoder",
        "aggregation_backend_name": "fedavg",
    }
    protocol["fl_method"]["manual_axes"]["update_family"] = "peft_text_encoder"
    protocol["objective"] = {
        "query_ssl.method_name": "fixmatch_usb_v1",
        "query_ssl.algorithm_name": "fixmatch",
        "peft_classifier.peft_adapter_name": "lora",
        "peft_classifier.rank": 8,
        "peft_classifier.alpha": 16,
        "peft_classifier.dropout": 0.1,
        "peft_classifier.bias": "none",
        "peft_classifier.target_modules": "all-linear",
        "peft_classifier.use_rslora": False,
        "peft_classifier.delta_format": "server_uploaded_artifact_ref",
    }
    report_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_legacy_entrypoint_report(tmp_path: Path) -> Path:
    report_path = (
        tmp_path
        / "runs"
        / "train_peft_ssl_classifier"
        / "consistency"
        / "legacy_run"
        / "reports"
        / "report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_sample_report(report_path.parent.parent), indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_smoke_report(tmp_path: Path) -> Path:
    report_path = (
        tmp_path
        / "runs"
        / "_smoke"
        / "run_peft_ssl_control"
        / "consistency"
        / "labeled-ourafla_reddit_unlabeled-ourafla_reddit"
        / "fixmatch_usb_v1"
        / "smoke_run"
        / "reports"
        / "report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}\n", encoding="utf-8")
    return report_path


def _write_metadata_smoke_report(tmp_path: Path) -> Path:
    report_path = (
        tmp_path
        / "runs"
        / "run_peft_ssl_control"
        / "consistency"
        / "metadata_smoke"
        / "reports"
        / "report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "central_peft_classifier_eval.v1",
                "trainer_version": "metadata_smoke",
                "manifest": {
                    "run_control": {
                        "budget_name": "smoke",
                        "output_root": "runs/_smoke",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_fl_ssl_sweep_report(tmp_path: Path, *, client_slug: str) -> Path:
    report_path = (
        tmp_path
        / "runs"
        / "federated_simulation_client_count_sweep_short"
        / "fixmatch_peft_adapter_alpha03_1round_20260518"
        / "20260517T193320Z"
        / client_slug
        / "reports"
        / "fl_ssl_main_comparison.report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_sample_fl_ssl_report(), indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_new_layout_fl_ssl_report(tmp_path: Path) -> Path:
    report_path = (
        tmp_path
        / "runs"
        / "fl_ssl"
        / "manual_baselines"
        / "fixmatch_usb_v1__peft_text_encoder_lora__fedavg"
        / "alpha03_seed42"
        / "clients10_rounds50"
        / "20260518T010203Z"
        / "reports"
        / "fl_ssl_main_comparison.report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_sample_fl_ssl_report(), indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def _sample_report(projection_dir: Path) -> dict:
    return {
        "schema_version": "central_peft_classifier_eval.v1",
        "trainer_version": "peft_fixmatch_2026_05_13_143419",
        "manifest": {
            "trainer_version": "peft_fixmatch_2026_05_13_143419",
            "train_jsonl": "data/datasets/source/labeled.jsonl",
            "eval_sets": {
                "validation": "data/datasets/source/validation.jsonl",
            },
            "selection_set": "validation",
            "run_control": {
                "track": "central_ssl",
                "budget_name": "main",
                "output_root": "runs",
            },
            "seed": 42,
            "epochs": 2,
            "max_train_steps": 100,
            "train_batch_size": 12,
            "eval_batch_size": 32,
            "learning_rate": 0.0002,
            "classifier_learning_rate": 0.0002,
            "categories": ["anxiety", "normal"],
            "backbone": {
                "peft_adapter_config": {
                    "adapter_name": "lora",
                    "rank": 8,
                    "alpha": 16,
                    "dropout": 0.1,
                    "bias": "none",
                    "target_modules": "all-linear",
                    "use_rslora": False,
                }
            },
            "history": [
                {
                    "epoch": 1,
                    "step": 2000,
                    "train_loss": 0.5,
                    "train_sup_loss": 0.4,
                    "train_unsup_loss": 0.1,
                    "train_util_ratio": 0.3,
                    "selection_loss": 0.6,
                    "selection_accuracy_top_1": 0.75,
                    "selection_macro_f1": 0.74,
                    "selection_expected_calibration_error": 0.11,
                    "selection_worst_category_f1": "normal",
                    "selection_worst_category_f1_value": 0.7,
                    "selection_per_category": {
                        "anxiety": {
                            "support": 2,
                            "predicted": 2,
                            "correct": 2,
                            "precision": 1.0,
                            "recall": 1.0,
                            "f1": 1.0,
                        },
                        "normal": {
                            "support": 2,
                            "predicted": 2,
                            "correct": 1,
                            "precision": 0.5,
                            "recall": 0.5,
                            "f1": 0.7,
                        },
                    },
                }
            ],
            "runtime_metrics": {
                "train_seconds": 10.0,
                "training_example_count": 100,
                "examples_per_second": 10.0,
                "trainable_param_ratio": 0.01,
            },
            "query_adaptation_initial_checkpoint": {
                "preset_name": "none",
            },
            "query_ssl_method": {
                "preset_name": "fixmatch_usb_v1",
                "algorithm_name": "fixmatch",
            },
            "projection_artifacts": {
                "enabled": True,
                "manifest_path": str(projection_dir / "projection_manifest.json"),
                "datasets": {
                    "validation": {
                        "reducer": "umap",
                        "fallback_reason": None,
                        "row_count": 4,
                        "points_jsonl": str(
                            projection_dir / "validation.projection.jsonl"
                        ),
                        "figure_png": str(projection_dir / "validation.projection.png"),
                    }
                },
            },
        },
        "results": {
            "validation": {
                "rows_total": 4,
                "loss": 0.5,
                "accuracy_top_1": 0.75,
                "correct_top_1": 3,
                "macro_precision": 0.8,
                "macro_recall": 0.77,
                "macro_f1": 0.78,
                "weighted_precision": 0.8,
                "weighted_recall": 0.75,
                "weighted_f1": 0.76,
                "balanced_accuracy": 0.77,
                "expected_calibration_error": 0.12,
                "max_calibration_error": 0.2,
                "overconfidence_gap": 0.1,
                "worst_category_f1": "normal",
                "worst_category_f1_value": 0.7,
                "mean_true_label_probability": 0.7,
                "mean_top_1_probability": 0.9,
                "mean_margin_top1_top2": 0.6,
                "confusion_matrix": {
                    "anxiety": {"anxiety": 2, "normal": 0},
                    "normal": {"anxiety": 1, "normal": 1},
                },
                "per_category": {
                    "anxiety": {
                        "support": 2,
                        "predicted": 3,
                        "correct": 2,
                        "precision": 0.66,
                        "recall": 1.0,
                        "f1": 0.8,
                        "mean_true_label_probability": 0.8,
                        "mean_top_1_probability": 0.9,
                        "mean_margin_top1_top2": 0.7,
                    },
                    "normal": {
                        "support": 2,
                        "predicted": 1,
                        "correct": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1": 0.7,
                        "mean_true_label_probability": 0.6,
                        "mean_top_1_probability": 0.85,
                        "mean_margin_top1_top2": 0.5,
                    },
                },
            }
        },
    }


def _sample_fl_ssl_report(projection_dir: Path | None = None) -> dict:
    validation_report = {
        "rows_total": 4,
        "loss": 0.5,
        "accuracy_top_1": 0.75,
        "correct_top_1": 3,
        "macro_precision": 0.8,
        "macro_recall": 0.77,
        "macro_f1": 0.78,
        "weighted_precision": 0.8,
        "weighted_recall": 0.75,
        "weighted_f1": 0.76,
        "balanced_accuracy": 0.77,
        "expected_calibration_error": 0.12,
        "max_calibration_error": 0.2,
        "overconfidence_gap": 0.1,
        "worst_category_f1": "normal",
        "worst_category_f1_value": 0.7,
        "mean_true_label_probability": 0.7,
        "mean_top_1_probability": 0.9,
        "mean_margin_top1_top2": 0.6,
        "confusion_matrix": {
            "anxiety": {"anxiety": 2, "normal": 0},
            "normal": {"anxiety": 1, "normal": 1},
        },
        "per_category": {
            "anxiety": {
                "support": 2,
                "predicted": 3,
                "correct": 2,
                "precision": 0.66,
                "recall": 1.0,
                "f1": 0.8,
                "mean_true_label_probability": 0.8,
                "mean_top_1_probability": 0.9,
                "mean_margin_top1_top2": 0.7,
            },
            "normal": {
                "support": 2,
                "predicted": 1,
                "correct": 1,
                "precision": 1.0,
                "recall": 0.5,
                "f1": 0.7,
                "mean_true_label_probability": 0.6,
                "mean_top_1_probability": 0.85,
                "mean_margin_top1_top2": 0.5,
            },
        },
    }
    initial_validation_report = {
        **validation_report,
        "loss": 0.6,
        "accuracy_top_1": 0.7,
        "macro_f1": 0.7,
        "expected_calibration_error": 0.2,
        "accepted_ratio": 0.4,
    }
    round_1_validation_report = {
        **validation_report,
        "loss": 0.55,
        "accuracy_top_1": 0.72,
        "macro_f1": 0.74,
        "expected_calibration_error": 0.16,
        "accepted_ratio": 0.45,
    }
    round_2_validation_report = {
        **validation_report,
        "accepted_ratio": 0.5,
    }
    payload = {
        "schema_version": "federated_simulation_report.v1",
        "track": "fl_ssl_main_comparison",
        "table_role": "paper_main",
        "must_not_merge_with": ["central_peft_ssl"],
        "protocol": {
            "client_count": 10,
            "round_budget": 50,
            "completed_rounds": 50,
            "seed": 42,
            "run_control": {
                "metadata_status": "recorded",
                "budget_name": "main",
                "output_dir": "runs/fl_ssl",
            },
            "shard_policy": {
                "name": "dirichlet_label_skew",
                "alpha": 0.3,
            },
            "fl_data_source": {
                "split_id": (
                    "labeled-ourafla_reddit_unlabeled-ourafla_reddit_"
                    "test-ourafla_reddit_"
                    "dirichlet_label_skew_clients10_seed42"
                ),
                "split_manifest_path": (
                    "data/datasets/fl_client_splits/main_alpha03_seed42/manifest.json"
                ),
                "source_selection": {
                    "labeled": "ourafla_reddit",
                    "unlabeled": "ourafla_reddit",
                    "validation": "ourafla_reddit",
                    "test": "ourafla_reddit",
                },
            },
            "labeled_unlabeled_split": {
                "actual_unlabeled_count": 40555,
                "actual_total_exposure_count": 40960,
                "actual_labeled_exposure_count": 405,
                "actual_unlabeled_exposure_count": 40555,
                "unique_total_count": 40596,
                "unique_labeled_count": 41,
                "unique_unlabeled_count": 40555,
            },
            "local_update_budget": {
                "local_epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.0001,
                "max_steps": 50,
            },
            "round_runtime": {
                "payload_adapter_kind": "peft_classifier",
                "update_family_name": "peft_text_encoder",
                "aggregation_backend_name": "fedavg",
            },
            "ssl_method": {
                "metadata_status": "not_applicable",
                "reason": "manual_composition",
            },
            "fl_method": {
                "name": "manual",
                "descriptor_name": None,
                "composition_mode": "manual",
                "execution_role": "manual_baseline",
                "manual_axes": {
                    "client_ssl_objective": "fixmatch",
                    "server_aggregation": "fedavg",
                    "update_family": "peft_text_encoder",
                },
            },
            "objective": {
                "query_ssl.method_name": "fixmatch_usb_v1",
                "query_ssl.algorithm_name": "fixmatch",
                "peft_classifier.peft_adapter_name": "lora",
                "peft_classifier.rank": 8,
                "peft_classifier.alpha": 16,
                "peft_classifier.dropout": 0.1,
                "peft_classifier.bias": "none",
                "peft_classifier.target_modules": "all-linear",
                "peft_classifier.use_rslora": False,
                "peft_classifier.delta_format": "server_uploaded_artifact_ref",
            },
            "embedding_adapter": {
                "backend": "transformers_mxbai",
                "model_id": "mixedbread-ai/mxbai-embed-large-v1",
                "device": "cuda",
            },
            "local_trainer_runtime": {
                "device": "cuda",
            },
        },
        "rounds": [
            {
                "round_id": "round_0001",
                "round_index": 1,
                "update_count": 10,
                "total_payload_bytes": 1000,
                "round_time_seconds": 12.5,
                "validation": round_1_validation_report,
                "global_validation": round_1_validation_report,
                "delta_from_initial": {
                    "loss_delta": -0.05,
                    "macro_f1_delta": 0.04,
                    "accuracy_top_1_delta": 0.02,
                    "expected_calibration_error_delta": -0.04,
                    "accepted_ratio_delta": 0.05,
                },
                "delta_from_previous_round": {
                    "loss_delta": -0.05,
                    "macro_f1_delta": 0.04,
                    "accuracy_top_1_delta": 0.02,
                    "expected_calibration_error_delta": -0.04,
                    "accepted_ratio_delta": 0.05,
                },
                "clients": [
                    {
                        "client_id": "agent_01",
                        "candidate_count": 12,
                        "accepted_count": 7,
                        "accepted_ratio": 0.5833333333,
                        "update_generated": True,
                        "aggregation_example_count": 20,
                        "delta_l2_norm": 1.1,
                        "client_train_time_seconds": 1.5,
                        "client_payload_bytes": 512,
                        "candidate_confidence_mean": 0.7,
                        "candidate_margin_mean": 0.2,
                        "pseudo_label_confidence_mean": 0.71,
                        "pseudo_label_margin_mean": 0.21,
                        "pseudo_label_accuracy": None,
                        "pseudo_label_correct_count": 0,
                        "pseudo_label_evaluated_count": 0,
                        "accepted_label_distribution": {"anxiety": 3, "normal": 4},
                        "rejected_label_distribution": {"normal": 5},
                    }
                ],
            },
            {
                "round_id": "round_0002",
                "round_index": 2,
                "update_count": 10,
                "total_payload_bytes": 1100,
                "round_time_seconds": 13.5,
                "validation": round_2_validation_report,
                "global_validation": round_2_validation_report,
                "delta_from_initial": {
                    "loss_delta": -0.1,
                    "macro_f1_delta": 0.08,
                    "accuracy_top_1_delta": 0.05,
                    "expected_calibration_error_delta": -0.08,
                    "accepted_ratio_delta": 0.1,
                },
                "delta_from_previous_round": {
                    "loss_delta": -0.05,
                    "macro_f1_delta": 0.04,
                    "accuracy_top_1_delta": 0.03,
                    "expected_calibration_error_delta": -0.04,
                    "accepted_ratio_delta": 0.05,
                },
                "clients": [
                    {
                        "client_id": "agent_01",
                        "candidate_count": 12,
                        "accepted_count": 8,
                        "accepted_ratio": 0.6666666667,
                        "update_generated": True,
                        "aggregation_example_count": 20,
                        "delta_l2_norm": 1.25,
                        "client_train_time_seconds": 1.6,
                        "client_payload_bytes": 520,
                        "candidate_confidence_mean": 0.72,
                        "candidate_margin_mean": 0.22,
                        "pseudo_label_confidence_mean": 0.73,
                        "pseudo_label_margin_mean": 0.23,
                        "pseudo_label_accuracy": None,
                        "pseudo_label_correct_count": 0,
                        "pseudo_label_evaluated_count": 0,
                        "accepted_label_distribution": {"anxiety": 4, "normal": 4},
                        "rejected_label_distribution": {"normal": 4},
                    }
                ],
            },
        ],
        "metrics": {
            "initial_validation": initial_validation_report,
            "final_validation": validation_report,
            "primary": {
                "macro_f1": 0.78,
                "worst_client_macro_f1": 0.41,
            },
            "secondary": {
                "loss": 0.5,
                "expected_calibration_error": 0.12,
                "communication_cost": {
                    "unit": "client_update_envelopes",
                    "value": 500,
                },
                "per_client_macro_f1_variance": 0.02,
            },
            "client_validation": {
                "evaluated_client_count": 10,
                "worst_client_macro_f1": 0.41,
                "best_client_macro_f1": 0.91,
                "macro_f1_std": 0.1,
                "loss_std": 0.2,
                "fairness_gap": 0.5,
                "clients": [
                    {
                        "client_id": "agent_01",
                        "client_train_size": 20,
                        "client_labeled_count": 8,
                        "client_unlabeled_count": 12,
                        "client_label_distribution": {
                            "anxiety": 8,
                            "normal": 12,
                        },
                        "client_candidate_count": 24,
                        "client_accepted_count": 15,
                        "client_accepted_ratio": 0.625,
                        "aggregation_example_count": 40,
                        "client_payload_bytes": 1032,
                        "client_update_generated": True,
                        "latest_round_id": "round_0002",
                        "latest_update_generated": True,
                        "update_generated_round_count": 2,
                        "client_delta_l2_norm": 1.25,
                        "mean_delta_l2_norm": 1.175,
                        "max_delta_l2_norm": 1.25,
                        "update_norm_variance": 0.005,
                        "client_train_time_seconds": 3.1,
                        "mean_client_train_time_seconds": 1.55,
                        "pseudo_label_accuracy": None,
                        "client_validation_loss": 0.9,
                        "client_validation_macro_f1": 0.41,
                        "client_validation_ece": 0.3,
                        "validation": {
                            **validation_report,
                            "rows_total": 20,
                            "loss": 0.9,
                            "accuracy_top_1": 0.5,
                            "macro_f1": 0.41,
                            "expected_calibration_error": 0.3,
                        },
                    }
                ],
            },
            "round_progression": {
                "validation_curve": [
                    {
                        "round_id": "initial",
                        "round_index": 0,
                        "macro_f1": 0.7,
                        "loss": 0.6,
                        "expected_calibration_error": 0.2,
                        "accepted_ratio": 0.4,
                        "accuracy_top_1": 0.7,
                    },
                    {
                        "round_id": "round_0001",
                        "round_index": 1,
                        "macro_f1": 0.74,
                        "loss": 0.55,
                        "expected_calibration_error": 0.16,
                        "accepted_ratio": 0.45,
                        "accuracy_top_1": 0.72,
                    },
                    {
                        "round_id": "round_0002",
                        "round_index": 2,
                        "macro_f1": 0.78,
                        "loss": 0.5,
                        "expected_calibration_error": 0.12,
                        "accepted_ratio": 0.5,
                        "accuracy_top_1": 0.75,
                    },
                ],
            },
        },
    }
    if projection_dir is not None:
        payload["diagnostics"] = {
            "final_projection_artifacts": {
                "enabled": True,
                "manifest_path": str(projection_dir / "projection_manifest.json"),
                "datasets": {
                    "validation": {
                        "reducer": "umap",
                        "fallback_reason": None,
                        "row_count": 4,
                        "points_jsonl": str(
                            projection_dir / "validation.projection.jsonl"
                        ),
                        "figure_png": str(projection_dir / "validation.projection.png"),
                    }
                },
            }
        }
    return payload

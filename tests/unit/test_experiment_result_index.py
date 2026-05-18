"""Experiment result index 검증."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.experiments.result_index.dashboard_export import write_dashboard_bundle
from scripts.experiments.result_index.report_loader import (
    discover_report_paths,
    load_result_index_records,
)
from scripts.experiments.result_index.sqlite_store import write_result_index_records


def test_load_result_index_records_normalizes_report_shape(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path)

    records = load_result_index_records(report_path)

    assert records.run.run_id == "lora_fixmatch_2026_05_13_143419"
    assert records.run.track == "central_lora_ssl"
    assert records.run.method_family == "lora_classifier"
    assert records.run.method_name == "fixmatch_usb_v1"
    assert records.run.algorithm_name == "fixmatch"
    assert records.run.labeled_dataset_name == "ourafla_reddit"
    assert records.run.unlabeled_dataset_name == "szegeelim_general4"
    assert records.run.validation_dataset_name == "ourafla_reddit"
    assert records.run.test_dataset_name == "ourafla_reddit"
    assert records.eval_metrics[0].macro_f1 == 0.78
    assert records.per_class_metrics[0].category == "anxiety"
    assert records.confusion_matrix_cells[0].actual_category == "anxiety"
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


def test_write_result_index_records_and_export_dashboard_json(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path)
    db_path = tmp_path / "experiment_results.sqlite"
    dashboard_path = tmp_path / "experiment_dashboard.json"
    records = load_result_index_records(report_path)

    write_result_index_records(db_path=db_path, records=[records])
    bundle = write_dashboard_bundle(db_path=db_path, output_path=dashboard_path)

    assert dashboard_path.exists()
    assert bundle["filters"]["methods"] == ["fixmatch_usb_v1"]
    assert bundle["runs"][0]["selection_slug"] == (
        "labeled-ourafla_reddit_unlabeled-szegeelim_general4_"
        "validation-ourafla_reddit_test-ourafla_reddit"
    )
    assert bundle["projection_images"][0]["image_src"].startswith(
        "data/artifacts/lora_fixmatch_2026_05_13_143419/"
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


def test_load_result_index_records_normalizes_fl_ssl_report_shape(
    tmp_path: Path,
) -> None:
    report_path = _write_fl_ssl_report(tmp_path)

    records = load_result_index_records(report_path)

    assert records.run.run_id == (
        "fixmatch_lora_alpha03_10c_50round_20260518__20260517T150549Z"
    )
    assert records.run.track == "fl_ssl_main_comparison"
    assert records.run.method_family == "lora_classifier"
    assert records.run.method_name == "fixmatch_usb_v1"
    assert records.run.algorithm_name == "fixmatch"
    assert records.run.selection_slug == (
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_"
        "validation-ourafla_reddit_test-ourafla_reddit_"
        "dirichlet_label_skew_clients10_seed42"
    )
    assert records.run.labeled_dataset_name == "ourafla_reddit"
    assert records.run.unlabeled_dataset_name == "ourafla_reddit"
    assert records.run.seed == 42
    assert records.run.client_count == 10
    assert records.run.round_budget == 50
    assert records.run.completed_rounds == 50
    assert records.run.shard_policy_name == "dirichlet_label_skew"
    assert records.run.shard_alpha == 0.3
    assert records.run.adapter_family_name == "lora_classifier"
    assert records.run.aggregation_backend_name == "fedavg"
    assert records.run.update_delta_format == "server_uploaded_artifact_ref"
    assert records.run.embedding_backend == "transformers_mxbai"
    assert records.run.embedding_device == "cuda"
    assert records.eval_metrics[1].eval_set == "final_validation"
    assert records.eval_metrics[1].macro_f1 == 0.78
    assert records.per_class_metrics[0].category == "anxiety"
    assert records.confusion_matrix_cells[0].actual_category == "anxiety"
    assert records.artifacts[0].artifact_kind == "fl_ssl_report"
    assert records.artifacts[1].artifact_kind == "fl_client_split_manifest"


def test_result_index_discovers_fl_ssl_report_artifacts(tmp_path: Path) -> None:
    central_report = _write_report(tmp_path)
    fl_report = _write_fl_ssl_report(tmp_path)

    assert discover_report_paths(tmp_path / "runs") == [fl_report, central_report]


def test_result_index_prefers_fl_ssl_hardlink_mirror(
    tmp_path: Path,
) -> None:
    legacy_report = _write_fl_ssl_report(tmp_path)
    mirrored_report = (
        tmp_path
        / "runs"
        / "fl_ssl"
        / "legacy"
        / "evidence"
        / "main"
        / "fixmatch_lora_alpha03_10c_50round_20260518"
        / "20260517T150549Z"
        / "reports"
        / "fl_ssl_main_comparison.report.json"
    )
    mirrored_report.parent.mkdir(parents=True, exist_ok=True)
    mirrored_report.hardlink_to(legacy_report)

    assert discover_report_paths(tmp_path / "runs") == [mirrored_report]


def test_load_result_index_records_keeps_client_count_sweep_slug(
    tmp_path: Path,
) -> None:
    report_path = _write_fl_ssl_sweep_report(tmp_path, client_slug="clients_03")

    records = load_result_index_records(report_path)

    assert records.run.run_id == (
        "fixmatch_lora_alpha03_1round_20260518__20260517T193320Z__clients_03"
    )


def test_load_result_index_records_keeps_new_fl_ssl_layout_parts(
    tmp_path: Path,
) -> None:
    report_path = _write_new_layout_fl_ssl_report(tmp_path)

    records = load_result_index_records(report_path)

    assert records.run.run_id == (
        "single__main__runtime_split_seed42_clients10_dirichlet_label_skew_alpha0p3__"
        "fixmatch_usb_v1_lora_classifier_fedavg_manual__20260518T010203Z"
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
    assert bundle["filters"]["client_counts"] == [10]
    assert bundle["filters"]["round_budgets"] == [50]
    assert bundle["filters"]["shard_alphas"] == [0.3]
    assert bundle["filters"]["adapter_families"] == ["lora_classifier"]
    assert bundle["filters"]["aggregation_backends"] == ["fedavg"]
    assert bundle["filters"]["update_delta_formats"] == ["server_uploaded_artifact_ref"]
    assert bundle["filters"]["embedding_backends"] == ["transformers_mxbai"]
    assert bundle["filters"]["embedding_model_ids"] == [
        "mixedbread-ai/mxbai-embed-large-v1"
    ]
    assert bundle["filters"]["embedding_devices"] == ["cuda"]
    assert bundle["filters"]["local_trainer_devices"] == ["cuda"]


def _write_report(tmp_path: Path) -> Path:
    report_path = (
        tmp_path
        / "runs"
        / "train_lora_ssl_classifier"
        / "consistency"
        / (
            "labeled-ourafla_reddit_unlabeled-szegeelim_general4_"
            "validation-ourafla_reddit_test-ourafla_reddit"
        )
        / "fixmatch_usb_v1"
        / "lora_fixmatch_2026_05_13_143419"
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


def _write_fl_ssl_report(tmp_path: Path) -> Path:
    report_path = (
        tmp_path
        / "runs"
        / "federated_simulation"
        / "fixmatch_lora_alpha03_10c_50round_20260518"
        / "20260517T150549Z"
        / "reports"
        / "fl_ssl_main_comparison.report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_sample_fl_ssl_report(), indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_fl_ssl_sweep_report(tmp_path: Path, *, client_slug: str) -> Path:
    report_path = (
        tmp_path
        / "runs"
        / "federated_simulation_client_count_sweep_short"
        / "fixmatch_lora_alpha03_1round_20260518"
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
        / "single"
        / "main"
        / "runtime_split_seed42_clients10_dirichlet_label_skew_alpha0p3"
        / "fixmatch_usb_v1_lora_classifier_fedavg_manual"
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
        "schema_version": "central_lora_classifier_eval.v1",
        "trainer_version": "lora_fixmatch_2026_05_13_143419",
        "manifest": {
            "trainer_version": "lora_fixmatch_2026_05_13_143419",
            "train_jsonl": "data/datasets/source/labeled.jsonl",
            "eval_sets": {
                "validation": "data/datasets/source/validation.jsonl",
            },
            "selection_set": "validation",
            "seed": 42,
            "epochs": 2,
            "max_train_steps": 100,
            "train_batch_size": 12,
            "eval_batch_size": 32,
            "learning_rate": 0.0002,
            "classifier_learning_rate": 0.0002,
            "categories": ["anxiety", "normal"],
            "history": [
                {
                    "epoch": 1,
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


def _sample_fl_ssl_report() -> dict:
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
    return {
        "schema_version": "federated_simulation_report.v1",
        "track": "fl_ssl_main_comparison",
        "table_role": "paper_main",
        "must_not_merge_with": ["central_lora_ssl"],
        "protocol": {
            "client_count": 10,
            "round_budget": 50,
            "completed_rounds": 50,
            "seed": 42,
            "shard_policy": {
                "name": "dirichlet_label_skew",
                "alpha": 0.3,
            },
            "fl_data_source": {
                "split_id": (
                    "labeled-ourafla_reddit_unlabeled-ourafla_reddit_"
                    "validation-ourafla_reddit_test-ourafla_reddit_"
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
            },
            "local_update_budget": {
                "local_epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.0001,
                "max_steps": 50,
            },
            "round_runtime": {
                "adapter_family_name": "lora_classifier",
                "aggregation_backend_name": "fedavg",
            },
            "ssl_method": {
                "name": "fedavg_pseudo_label",
            },
            "objective": {
                "query_ssl.method_name": "fixmatch_usb_v1",
                "query_ssl.algorithm_name": "fixmatch",
                "lora_classifier.delta_format": "server_uploaded_artifact_ref",
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
        "metrics": {
            "initial_validation": validation_report,
            "final_validation": validation_report,
        },
    }

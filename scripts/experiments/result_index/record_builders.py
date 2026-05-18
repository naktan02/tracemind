"""Build normalized result index child records from report payload sections."""

from __future__ import annotations

from typing import Any

from scripts.experiments.result_index.models import (
    ArtifactRecord,
    ConfusionMatrixCellRecord,
    EpochPerClassMetricRecord,
    EvalMetricRecord,
    PerClassMetricRecord,
)
from scripts.experiments.result_index.report_parsing import (
    as_mapping,
    optional_float,
    optional_int,
    optional_str,
)


def build_eval_metric(
    *,
    run_id: str,
    eval_set: str,
    report: dict[str, Any],
) -> EvalMetricRecord:
    return EvalMetricRecord(
        run_id=run_id,
        eval_set=eval_set,
        rows_total=optional_int(report.get("rows_total")),
        loss=optional_float(report.get("loss")),
        accuracy_top_1=optional_float(report.get("accuracy_top_1")),
        macro_precision=optional_float(report.get("macro_precision")),
        macro_recall=optional_float(report.get("macro_recall")),
        macro_f1=optional_float(report.get("macro_f1")),
        weighted_precision=optional_float(report.get("weighted_precision")),
        weighted_recall=optional_float(report.get("weighted_recall")),
        weighted_f1=optional_float(report.get("weighted_f1")),
        balanced_accuracy=optional_float(report.get("balanced_accuracy")),
        expected_calibration_error=optional_float(
            report.get("expected_calibration_error")
        ),
        max_calibration_error=optional_float(report.get("max_calibration_error")),
        overconfidence_gap=optional_float(report.get("overconfidence_gap")),
        worst_category_f1=optional_str(report.get("worst_category_f1")),
        worst_category_f1_value=optional_float(report.get("worst_category_f1_value")),
        worst_category_precision=optional_float(report.get("worst_category_precision")),
        worst_category_recall=optional_float(report.get("worst_category_recall")),
        mean_true_label_probability=optional_float(
            report.get("mean_true_label_probability")
        ),
        mean_top_1_probability=optional_float(report.get("mean_top_1_probability")),
        mean_margin_top1_top2=optional_float(report.get("mean_margin_top1_top2")),
        correct_top_1=optional_int(report.get("correct_top_1")),
    )


def build_per_class_metrics(
    *,
    run_id: str,
    eval_set: str,
    per_category: dict[str, Any],
) -> list[PerClassMetricRecord]:
    records: list[PerClassMetricRecord] = []
    for category, metrics in sorted(per_category.items()):
        if not isinstance(metrics, dict):
            continue
        records.append(
            PerClassMetricRecord(
                run_id=run_id,
                eval_set=eval_set,
                category=str(category),
                support=optional_int(metrics.get("support")),
                predicted=optional_int(metrics.get("predicted")),
                correct=optional_int(metrics.get("correct")),
                precision=optional_float(metrics.get("precision")),
                recall=optional_float(metrics.get("recall")),
                f1=optional_float(metrics.get("f1")),
                mean_true_label_probability=optional_float(
                    metrics.get("mean_true_label_probability")
                ),
                mean_top_1_probability=optional_float(
                    metrics.get("mean_top_1_probability")
                ),
                mean_margin_top1_top2=optional_float(
                    metrics.get("mean_margin_top1_top2")
                ),
            )
        )
    return records


def build_confusion_matrix_cells(
    *,
    run_id: str,
    eval_set: str,
    confusion_matrix: dict[str, Any],
) -> list[ConfusionMatrixCellRecord]:
    records: list[ConfusionMatrixCellRecord] = []
    for actual_category, row in sorted(confusion_matrix.items()):
        if not isinstance(row, dict):
            continue
        for predicted_category, count in sorted(row.items()):
            records.append(
                ConfusionMatrixCellRecord(
                    run_id=run_id,
                    eval_set=eval_set,
                    actual_category=str(actual_category),
                    predicted_category=str(predicted_category),
                    count=int(count),
                )
            )
    return records


def build_epoch_per_class_metrics(
    *,
    run_id: str,
    epoch: int,
    per_category: dict[str, Any],
) -> list[EpochPerClassMetricRecord]:
    records: list[EpochPerClassMetricRecord] = []
    for category, metrics in sorted(per_category.items()):
        if not isinstance(metrics, dict):
            continue
        records.append(
            EpochPerClassMetricRecord(
                run_id=run_id,
                epoch=epoch,
                category=str(category),
                support=optional_int(metrics.get("support")),
                predicted=optional_int(metrics.get("predicted")),
                correct=optional_int(metrics.get("correct")),
                precision=optional_float(metrics.get("precision")),
                recall=optional_float(metrics.get("recall")),
                f1=optional_float(metrics.get("f1")),
                mean_true_label_probability=optional_float(
                    metrics.get("mean_true_label_probability")
                ),
                mean_top_1_probability=optional_float(
                    metrics.get("mean_top_1_probability")
                ),
                mean_margin_top1_top2=optional_float(
                    metrics.get("mean_margin_top1_top2")
                ),
            )
        )
    return records


def build_projection_artifacts(
    *,
    run_id: str,
    projection_artifacts: dict[str, Any],
) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    manifest_path = optional_str(projection_artifacts.get("manifest_path"))
    if manifest_path:
        records.append(
            ArtifactRecord(
                run_id=run_id,
                eval_set=None,
                artifact_kind="projection_manifest",
                artifact_ref=manifest_path,
                reducer=None,
                fallback_reason=None,
            )
        )
    for eval_set, entry in as_mapping(projection_artifacts.get("datasets")).items():
        if not isinstance(entry, dict):
            continue
        reducer = optional_str(entry.get("reducer"))
        fallback_reason = optional_str(entry.get("fallback_reason"))
        points_jsonl = optional_str(entry.get("points_jsonl"))
        figure_png = optional_str(entry.get("figure_png"))
        if points_jsonl:
            records.append(
                ArtifactRecord(
                    run_id=run_id,
                    eval_set=str(eval_set),
                    artifact_kind="projection_points_jsonl",
                    artifact_ref=points_jsonl,
                    reducer=reducer,
                    fallback_reason=fallback_reason,
                )
            )
        if figure_png:
            records.append(
                ArtifactRecord(
                    run_id=run_id,
                    eval_set=str(eval_set),
                    artifact_kind="projection_png",
                    artifact_ref=figure_png,
                    reducer=reducer,
                    fallback_reason=fallback_reason,
                )
            )
    return records

"""Load experiment reports into normalized result index records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.experiments.result_index.models import (
    ConfusionMatrixCellRecord,
    EpochMetricRecord,
    EpochPerClassMetricRecord,
    EvalMetricRecord,
    ExperimentRunRecord,
    PerClassMetricRecord,
    ResultIndexRecords,
)

_RUN_TIMESTAMP_RE = re.compile(
    r"(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_(?P<hms>\d{6})"
)


def discover_report_paths(runs_root: Path) -> list[Path]:
    """Find canonical report.json files under a runs root."""

    if runs_root.is_file():
        return [runs_root]
    return sorted(
        path
        for path in runs_root.rglob("report.json")
        if path.is_file() and path.parent.name == "reports"
    )


def load_result_index_records(report_path: Path) -> ResultIndexRecords:
    """Parse one report JSON into DB-ready records."""

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Report must be a JSON object: {report_path}")

    manifest = _as_mapping(payload.get("manifest"))
    results = _as_mapping(payload.get("results"))
    trainer_version = str(
        payload.get("trainer_version") or manifest.get("trainer_version") or ""
    ).strip()
    if not trainer_version:
        raise ValueError(f"Report is missing trainer_version: {report_path}")

    selection_slug = _find_selection_slug(report_path)
    split_names = _parse_selection_slug(selection_slug)
    query_ssl_method = _as_mapping(manifest.get("query_ssl_method"))
    runtime_metrics = _as_mapping(manifest.get("runtime_metrics"))
    initial_checkpoint = _as_mapping(
        manifest.get("query_adaptation_initial_checkpoint")
    )

    run = ExperimentRunRecord(
        run_id=trainer_version,
        track=_infer_track(report_path=report_path, payload=payload),
        method_family=_infer_method_family(payload),
        method_name=_infer_method_name(
            report_path=report_path,
            query_ssl_method=query_ssl_method,
        ),
        algorithm_name=_optional_str(query_ssl_method.get("algorithm_name")),
        selection_slug=selection_slug,
        labeled_dataset_name=split_names.get("labeled"),
        unlabeled_dataset_name=split_names.get("unlabeled"),
        validation_dataset_name=split_names.get("validation"),
        test_dataset_name=split_names.get("test"),
        seed=_optional_int(manifest.get("seed")),
        learning_rate=_optional_float(manifest.get("learning_rate")),
        classifier_learning_rate=_optional_float(
            manifest.get("classifier_learning_rate")
        ),
        epochs=_optional_int(manifest.get("epochs")),
        max_train_steps=_optional_int(manifest.get("max_train_steps")),
        train_batch_size=_optional_int(manifest.get("train_batch_size")),
        eval_batch_size=_optional_int(manifest.get("eval_batch_size")),
        initial_checkpoint_name=_optional_str(
            initial_checkpoint.get("preset_name")
            or initial_checkpoint.get("resolved_kind")
            or initial_checkpoint.get("mode")
        ),
        unlabeled_row_count=_optional_int(manifest.get("unlabeled_row_count")),
        train_seconds=_optional_float(runtime_metrics.get("train_seconds")),
        training_example_count=_optional_int(
            runtime_metrics.get("training_example_count")
        ),
        examples_per_second=_optional_float(runtime_metrics.get("examples_per_second")),
        trainable_param_ratio=_optional_float(
            runtime_metrics.get("trainable_param_ratio")
        ),
        created_at=_infer_created_at(trainer_version),
    )

    eval_metrics: list[EvalMetricRecord] = []
    per_class_metrics: list[PerClassMetricRecord] = []
    confusion_matrix_cells: list[ConfusionMatrixCellRecord] = []
    for eval_set, report in results.items():
        if not isinstance(eval_set, str) or not isinstance(report, dict):
            continue
        eval_metrics.append(
            _build_eval_metric(run_id=trainer_version, eval_set=eval_set, report=report)
        )
        per_class_metrics.extend(
            _build_per_class_metrics(
                run_id=trainer_version,
                eval_set=eval_set,
                per_category=_as_mapping(report.get("per_category")),
            )
        )
        confusion_matrix_cells.extend(
            _build_confusion_matrix_cells(
                run_id=trainer_version,
                eval_set=eval_set,
                confusion_matrix=_as_mapping(report.get("confusion_matrix")),
            )
        )

    epoch_metrics: list[EpochMetricRecord] = []
    epoch_per_class_metrics: list[EpochPerClassMetricRecord] = []
    for history_record in _as_sequence(manifest.get("history")):
        if not isinstance(history_record, dict):
            continue
        epoch = _optional_int(history_record.get("epoch"))
        if epoch is None:
            continue
        epoch_metrics.append(
            EpochMetricRecord(
                run_id=trainer_version,
                epoch=epoch,
                train_loss=_optional_float(history_record.get("train_loss")),
                train_sup_loss=_optional_float(history_record.get("train_sup_loss")),
                train_unsup_loss=_optional_float(
                    history_record.get("train_unsup_loss")
                ),
                train_util_ratio=_optional_float(
                    history_record.get("train_util_ratio")
                ),
                selection_loss=_optional_float(history_record.get("selection_loss")),
                selection_accuracy_top_1=_optional_float(
                    history_record.get("selection_accuracy_top_1")
                ),
                selection_macro_f1=_optional_float(
                    history_record.get("selection_macro_f1")
                ),
                selection_expected_calibration_error=_optional_float(
                    history_record.get("selection_expected_calibration_error")
                ),
                selection_worst_category_f1=_optional_str(
                    history_record.get("selection_worst_category_f1")
                ),
                selection_worst_category_f1_value=_optional_float(
                    history_record.get("selection_worst_category_f1_value")
                ),
            )
        )
        epoch_per_class_metrics.extend(
            _build_epoch_per_class_metrics(
                run_id=trainer_version,
                epoch=epoch,
                per_category=_as_mapping(history_record.get("selection_per_category")),
            )
        )

    return ResultIndexRecords(
        run=run,
        eval_metrics=tuple(eval_metrics),
        per_class_metrics=tuple(per_class_metrics),
        confusion_matrix_cells=tuple(confusion_matrix_cells),
        epoch_metrics=tuple(epoch_metrics),
        epoch_per_class_metrics=tuple(epoch_per_class_metrics),
    )


def _build_eval_metric(
    *,
    run_id: str,
    eval_set: str,
    report: dict[str, Any],
) -> EvalMetricRecord:
    return EvalMetricRecord(
        run_id=run_id,
        eval_set=eval_set,
        rows_total=_optional_int(report.get("rows_total")),
        loss=_optional_float(report.get("loss")),
        accuracy_top_1=_optional_float(report.get("accuracy_top_1")),
        macro_precision=_optional_float(report.get("macro_precision")),
        macro_recall=_optional_float(report.get("macro_recall")),
        macro_f1=_optional_float(report.get("macro_f1")),
        weighted_precision=_optional_float(report.get("weighted_precision")),
        weighted_recall=_optional_float(report.get("weighted_recall")),
        weighted_f1=_optional_float(report.get("weighted_f1")),
        balanced_accuracy=_optional_float(report.get("balanced_accuracy")),
        expected_calibration_error=_optional_float(
            report.get("expected_calibration_error")
        ),
        max_calibration_error=_optional_float(report.get("max_calibration_error")),
        overconfidence_gap=_optional_float(report.get("overconfidence_gap")),
        worst_category_f1=_optional_str(report.get("worst_category_f1")),
        worst_category_f1_value=_optional_float(report.get("worst_category_f1_value")),
        worst_category_precision=_optional_float(
            report.get("worst_category_precision")
        ),
        worst_category_recall=_optional_float(report.get("worst_category_recall")),
        mean_true_label_probability=_optional_float(
            report.get("mean_true_label_probability")
        ),
        mean_top_1_probability=_optional_float(report.get("mean_top_1_probability")),
        mean_margin_top1_top2=_optional_float(report.get("mean_margin_top1_top2")),
        correct_top_1=_optional_int(report.get("correct_top_1")),
    )


def _build_per_class_metrics(
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
                support=_optional_int(metrics.get("support")),
                predicted=_optional_int(metrics.get("predicted")),
                correct=_optional_int(metrics.get("correct")),
                precision=_optional_float(metrics.get("precision")),
                recall=_optional_float(metrics.get("recall")),
                f1=_optional_float(metrics.get("f1")),
                mean_true_label_probability=_optional_float(
                    metrics.get("mean_true_label_probability")
                ),
                mean_top_1_probability=_optional_float(
                    metrics.get("mean_top_1_probability")
                ),
                mean_margin_top1_top2=_optional_float(
                    metrics.get("mean_margin_top1_top2")
                ),
            )
        )
    return records


def _build_confusion_matrix_cells(
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


def _build_epoch_per_class_metrics(
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
                support=_optional_int(metrics.get("support")),
                predicted=_optional_int(metrics.get("predicted")),
                correct=_optional_int(metrics.get("correct")),
                precision=_optional_float(metrics.get("precision")),
                recall=_optional_float(metrics.get("recall")),
                f1=_optional_float(metrics.get("f1")),
                mean_true_label_probability=_optional_float(
                    metrics.get("mean_true_label_probability")
                ),
                mean_top_1_probability=_optional_float(
                    metrics.get("mean_top_1_probability")
                ),
                mean_margin_top1_top2=_optional_float(
                    metrics.get("mean_margin_top1_top2")
                ),
            )
        )
    return records


def _find_selection_slug(report_path: Path) -> str | None:
    for parent in report_path.parents:
        name = parent.name
        if name.startswith("labeled-") and "_unlabeled-" in name:
            return name
    return None


def _parse_selection_slug(selection_slug: str | None) -> dict[str, str | None]:
    names = {
        "labeled": None,
        "unlabeled": None,
        "validation": None,
        "test": None,
    }
    if not selection_slug:
        return names
    try:
        labeled_part, rest = selection_slug.removeprefix("labeled-").split(
            "_unlabeled-",
            1,
        )
        unlabeled_part, rest = rest.split("_validation-", 1)
        validation_part, test_part = rest.split("_test-", 1)
    except ValueError:
        return names
    names.update(
        {
            "labeled": labeled_part or None,
            "unlabeled": unlabeled_part or None,
            "validation": validation_part or None,
            "test": test_part or None,
        }
    )
    return names


def _infer_track(*, report_path: Path, payload: dict[str, Any]) -> str:
    parts = set(report_path.parts)
    if "train_lora_ssl_classifier" in parts:
        return "central_lora_ssl"
    if "train_lora_supervised_classifier" in parts:
        return "central_lora_supervised"
    if "train_classifier" in parts:
        return "central_classifier_seed"
    schema_version = str(payload.get("schema_version") or "").strip()
    return schema_version or "unknown"


def _infer_method_family(payload: dict[str, Any]) -> str:
    schema_version = str(payload.get("schema_version") or "")
    if schema_version == "central_lora_classifier_eval.v1":
        return "lora_classifier"
    return "unknown"


def _infer_method_name(
    *,
    report_path: Path,
    query_ssl_method: dict[str, Any],
) -> str:
    preset_name = _optional_str(
        query_ssl_method.get("preset_name") or query_ssl_method.get("name")
    )
    if preset_name:
        return preset_name
    run_dir = report_path.parent.parent
    parent_name = run_dir.parent.name
    if parent_name and not parent_name.startswith("labeled-"):
        return parent_name
    return "supervised"


def _infer_created_at(run_id: str) -> str | None:
    match = _RUN_TIMESTAMP_RE.search(run_id)
    if match is None:
        return None
    hms = match.group("hms")
    return (
        f"{match.group('year')}-{match.group('month')}-{match.group('day')}"
        f"T{hms[0:2]}:{hms[2:4]}:{hms[4:6]}"
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

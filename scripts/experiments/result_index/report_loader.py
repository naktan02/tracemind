"""Load experiment reports into normalized result index records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.experiments.result_index.models import (
    ArtifactRecord,
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
    """Find canonical experiment report files under a runs root."""

    if runs_root.is_file():
        return [runs_root]
    report_names = {"report.json", "fl_ssl_main_comparison.report.json"}
    return sorted(
        path
        for path in runs_root.rglob("*.json")
        if path.is_file()
        and path.parent.name == "reports"
        and path.name in report_names
    )


def load_result_index_records(report_path: Path) -> ResultIndexRecords:
    """Parse one report JSON into DB-ready records."""

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Report must be a JSON object: {report_path}")
    if _is_fl_ssl_report(payload):
        return _load_fl_ssl_result_index_records(
            report_path=report_path,
            payload=payload,
        )

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
        client_count=None,
        round_budget=None,
        completed_rounds=None,
        shard_policy_name=None,
        shard_alpha=None,
        adapter_family_name=None,
        aggregation_backend_name=None,
        update_delta_format=None,
        embedding_backend=None,
        embedding_model_id=None,
        embedding_device=None,
        local_trainer_device=None,
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

    artifacts = _build_artifacts(
        run_id=trainer_version,
        projection_artifacts=_as_mapping(manifest.get("projection_artifacts")),
    )

    return ResultIndexRecords(
        run=run,
        eval_metrics=tuple(eval_metrics),
        per_class_metrics=tuple(per_class_metrics),
        confusion_matrix_cells=tuple(confusion_matrix_cells),
        epoch_metrics=tuple(epoch_metrics),
        epoch_per_class_metrics=tuple(epoch_per_class_metrics),
        artifacts=tuple(artifacts),
    )


def _load_fl_ssl_result_index_records(
    *,
    report_path: Path,
    payload: dict[str, Any],
) -> ResultIndexRecords:
    protocol = _as_mapping(payload.get("protocol"))
    metrics = _as_mapping(payload.get("metrics"))
    fl_data_source = _as_mapping(protocol.get("fl_data_source"))
    source_selection = _as_mapping(fl_data_source.get("source_selection"))
    round_runtime = _as_mapping(protocol.get("round_runtime"))
    objective = _as_mapping(protocol.get("objective"))
    shard_policy = _as_mapping(protocol.get("shard_policy"))
    local_update_budget = _as_mapping(protocol.get("local_update_budget"))
    embedding_adapter = _as_mapping(protocol.get("embedding_adapter"))
    local_trainer_runtime = _as_mapping(protocol.get("local_trainer_runtime"))

    run_id = _infer_fl_ssl_run_id(report_path)
    method_name = (
        _optional_str(objective.get("query_ssl.method_name"))
        or _optional_str(_as_mapping(protocol.get("ssl_method")).get("name"))
        or "unknown"
    )
    adapter_family = _optional_str(round_runtime.get("adapter_family_name"))
    run = ExperimentRunRecord(
        run_id=run_id,
        track=_optional_str(payload.get("track")) or "fl_ssl_main_comparison",
        method_family=adapter_family or "unknown",
        method_name=method_name,
        algorithm_name=_optional_str(objective.get("query_ssl.algorithm_name")),
        selection_slug=_optional_str(fl_data_source.get("split_id")),
        labeled_dataset_name=_optional_str(source_selection.get("labeled")),
        unlabeled_dataset_name=_optional_str(source_selection.get("unlabeled")),
        validation_dataset_name=_optional_str(source_selection.get("validation")),
        test_dataset_name=_optional_str(source_selection.get("test")),
        seed=_optional_int(protocol.get("seed")),
        learning_rate=_optional_float(local_update_budget.get("learning_rate")),
        classifier_learning_rate=None,
        epochs=_optional_int(local_update_budget.get("local_epochs")),
        max_train_steps=_optional_int(local_update_budget.get("max_steps")),
        train_batch_size=_optional_int(local_update_budget.get("batch_size")),
        eval_batch_size=None,
        initial_checkpoint_name=None,
        unlabeled_row_count=_optional_int(
            _as_mapping(protocol.get("labeled_unlabeled_split")).get(
                "actual_unlabeled_count"
            )
        ),
        train_seconds=None,
        training_example_count=None,
        examples_per_second=None,
        trainable_param_ratio=None,
        client_count=_optional_int(protocol.get("client_count")),
        round_budget=_optional_int(protocol.get("round_budget")),
        completed_rounds=_optional_int(protocol.get("completed_rounds")),
        shard_policy_name=_optional_str(shard_policy.get("name")),
        shard_alpha=_optional_float(shard_policy.get("alpha")),
        adapter_family_name=adapter_family,
        aggregation_backend_name=_optional_str(
            round_runtime.get("aggregation_backend_name")
        ),
        update_delta_format=_optional_str(
            objective.get("lora_classifier.delta_format")
        ),
        embedding_backend=_optional_str(embedding_adapter.get("backend")),
        embedding_model_id=_optional_str(embedding_adapter.get("model_id")),
        embedding_device=_optional_str(embedding_adapter.get("device")),
        local_trainer_device=_optional_str(local_trainer_runtime.get("device")),
        created_at=_infer_created_at(run_id),
    )

    eval_metrics: list[EvalMetricRecord] = []
    per_class_metrics: list[PerClassMetricRecord] = []
    confusion_matrix_cells: list[ConfusionMatrixCellRecord] = []
    for eval_set in ("initial_validation", "final_validation"):
        report = _as_mapping(metrics.get(eval_set))
        if not report:
            continue
        eval_metrics.append(
            _build_eval_metric(run_id=run_id, eval_set=eval_set, report=report)
        )
        per_class_metrics.extend(
            _build_per_class_metrics(
                run_id=run_id,
                eval_set=eval_set,
                per_category=_as_mapping(report.get("per_category")),
            )
        )
        confusion_matrix_cells.extend(
            _build_confusion_matrix_cells(
                run_id=run_id,
                eval_set=eval_set,
                confusion_matrix=_as_mapping(report.get("confusion_matrix")),
            )
        )

    artifacts = [
        ArtifactRecord(
            run_id=run_id,
            eval_set=None,
            artifact_kind="fl_ssl_report",
            artifact_ref=str(report_path),
            reducer=None,
            fallback_reason=None,
        )
    ]
    split_manifest_path = _optional_str(fl_data_source.get("split_manifest_path"))
    if split_manifest_path:
        artifacts.append(
            ArtifactRecord(
                run_id=run_id,
                eval_set=None,
                artifact_kind="fl_client_split_manifest",
                artifact_ref=split_manifest_path,
                reducer=None,
                fallback_reason=None,
            )
        )

    return ResultIndexRecords(
        run=run,
        eval_metrics=tuple(eval_metrics),
        per_class_metrics=tuple(per_class_metrics),
        confusion_matrix_cells=tuple(confusion_matrix_cells),
        epoch_metrics=(),
        epoch_per_class_metrics=(),
        artifacts=tuple(artifacts),
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


def _build_artifacts(
    *,
    run_id: str,
    projection_artifacts: dict[str, Any],
) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    manifest_path = _optional_str(projection_artifacts.get("manifest_path"))
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
    for eval_set, entry in _as_mapping(projection_artifacts.get("datasets")).items():
        if not isinstance(entry, dict):
            continue
        reducer = _optional_str(entry.get("reducer"))
        fallback_reason = _optional_str(entry.get("fallback_reason"))
        points_jsonl = _optional_str(entry.get("points_jsonl"))
        figure_png = _optional_str(entry.get("figure_png"))
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


def _is_fl_ssl_report(payload: dict[str, Any]) -> bool:
    return (
        str(payload.get("track") or "") == "fl_ssl_main_comparison"
        or str(payload.get("schema_version") or "") == "federated_simulation_report.v1"
    )


def _infer_fl_ssl_run_id(report_path: Path) -> str:
    run_dir = report_path.parent.parent
    if run_dir.name.startswith("clients_"):
        run_timestamp_dir = run_dir.parent
        run_group = run_timestamp_dir.parent.name
        if run_group:
            return f"{run_group}__{run_timestamp_dir.name}__{run_dir.name}"
    run_group = run_dir.parent.name
    if run_group:
        return f"{run_group}__{run_dir.name}"
    return run_dir.name


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
    if match is not None:
        hms = match.group("hms")
        return (
            f"{match.group('year')}-{match.group('month')}-{match.group('day')}"
            f"T{hms[0:2]}:{hms[2:4]}:{hms[4:6]}"
        )
    compact_match = re.search(
        r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})T"
        r"(?P<hms>\d{6})Z",
        run_id,
    )
    if compact_match is None:
        return None
    hms = compact_match.group("hms")
    return (
        f"{compact_match.group('year')}-{compact_match.group('month')}-"
        f"{compact_match.group('day')}"
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

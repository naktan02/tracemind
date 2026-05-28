"""Load experiment reports into normalized result index records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.experiments.result_index.fl_ssl_report_loader import (
    is_fl_ssl_report,
    load_fl_ssl_result_index_records,
)
from scripts.experiments.result_index.models import (
    ConfusionMatrixCellRecord,
    EpochMetricRecord,
    EpochPerClassMetricRecord,
    EvalMetricRecord,
    ExperimentRunRecord,
    PerClassMetricRecord,
    ResultIndexRecords,
)
from scripts.experiments.result_index.record_builders import (
    build_confusion_matrix_cells,
    build_epoch_per_class_metrics,
    build_eval_metric,
    build_per_class_metrics,
    build_projection_artifacts,
)
from scripts.experiments.result_index.report_parsing import (
    as_mapping,
    as_sequence,
    infer_created_at,
    json_object_snapshot,
    optional_float,
    optional_int,
    optional_str,
)

CENTRAL_PEFT_SSL_CONTROL_PATH_NAMES = frozenset(
    {
        "run_peft_ssl_control",
        "train_peft_ssl_classifier",
    }
)
CENTRAL_PEFT_SUPERVISED_CONTROL_PATH_NAMES = frozenset(
    {
        "run_peft_supervised_control",
        "train_peft_supervised_classifier",
    }
)


def discover_report_paths(runs_root: Path) -> list[Path]:
    """Find canonical experiment report files under a runs root."""

    if runs_root.is_file():
        return [runs_root]
    report_names = {
        "report.json",
        "fl_ssl_main_comparison.report.json",
        "initial_eval.report.json",
    }
    exclude_smoke = runs_root.name == "runs"
    report_paths = sorted(
        path
        for path in runs_root.rglob("*.json")
        if path.is_file()
        and path.name in report_names
        and (path.parent.name == "reports" or path.name == "initial_eval.report.json")
        and not _is_default_excluded_smoke_path(
            runs_root=runs_root,
            path=path,
            exclude_smoke=exclude_smoke,
        )
    )
    return _deduplicate_hardlinked_report_paths(report_paths)


def _is_default_excluded_smoke_path(
    *,
    runs_root: Path,
    path: Path,
    exclude_smoke: bool,
) -> bool:
    """기본 `runs` ingest에서는 smoke 산출물을 웹/index에서 제외한다."""

    if not exclude_smoke:
        return False

    try:
        relative_parts = path.relative_to(runs_root).parts
    except ValueError:
        relative_parts = path.parts
    if "_smoke" in relative_parts:
        return True
    return _report_payload_budget_name(path) == "smoke"


def _report_payload_budget_name(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    protocol = as_mapping(payload.get("protocol"))
    run_control = as_mapping(protocol.get("run_control")) or as_mapping(
        as_mapping(payload.get("manifest")).get("run_control")
    )
    budget_name = optional_str(run_control.get("budget_name"))
    return None if budget_name is None else budget_name.strip().lower()


def _deduplicate_hardlinked_report_paths(report_paths: list[Path]) -> list[Path]:
    """같은 artifact가 여러 경로에 있으면 canonical `fl_ssl` 경로를 우선한다."""

    selected: dict[tuple[object, ...], Path] = {}
    for path in report_paths:
        identity = _report_file_identity(path)
        previous = selected.get(identity)
        if previous is None or _report_path_preference(path) < (
            _report_path_preference(previous)
        ):
            selected[identity] = path
    return sorted(selected.values())


def _report_file_identity(path: Path) -> tuple[object, ...]:
    try:
        stat_result = path.stat()
    except OSError:
        return ("path", str(path))
    return ("inode", stat_result.st_dev, stat_result.st_ino)


def _report_path_preference(path: Path) -> int:
    return 0 if "fl_ssl" in path.parts else 1


def load_result_index_records(report_path: Path) -> ResultIndexRecords:
    """Parse one report JSON into DB-ready records."""

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Report must be a JSON object: {report_path}")
    if is_fl_ssl_report(payload):
        return load_fl_ssl_result_index_records(
            report_path=report_path,
            payload=payload,
        )

    manifest = as_mapping(payload.get("manifest"))
    results = as_mapping(payload.get("results"))
    trainer_version = str(
        payload.get("trainer_version") or manifest.get("trainer_version") or ""
    ).strip()
    if not trainer_version:
        raise ValueError(f"Report is missing trainer_version: {report_path}")

    selection_slug = _find_selection_slug(report_path)
    split_names = _parse_selection_slug(selection_slug)
    query_ssl_method = as_mapping(manifest.get("query_ssl_method"))
    runtime_metrics = as_mapping(manifest.get("runtime_metrics"))
    initial_checkpoint = as_mapping(manifest.get("query_adaptation_initial_checkpoint"))
    run_control = as_mapping(manifest.get("run_control"))
    peft_adapter_config = _peft_adapter_config_from_backbone(
        as_mapping(manifest.get("backbone"))
    )

    run = ExperimentRunRecord(
        run_id=trainer_version,
        track=_infer_track(report_path=report_path, payload=payload),
        method_family=_infer_method_family(payload),
        method_name=_infer_method_name(
            report_path=report_path,
            query_ssl_method=query_ssl_method,
        ),
        algorithm_name=optional_str(query_ssl_method.get("algorithm_name")),
        selection_slug=selection_slug,
        labeled_dataset_name=split_names.get("labeled"),
        unlabeled_dataset_name=split_names.get("unlabeled"),
        validation_dataset_name=split_names.get("validation"),
        test_dataset_name=split_names.get("test"),
        seed=optional_int(manifest.get("seed")),
        learning_rate=optional_float(manifest.get("learning_rate")),
        classifier_learning_rate=optional_float(
            manifest.get("classifier_learning_rate")
        ),
        epochs=optional_int(manifest.get("epochs")),
        max_train_steps=optional_int(manifest.get("max_train_steps")),
        train_batch_size=optional_int(manifest.get("train_batch_size")),
        eval_batch_size=optional_int(manifest.get("eval_batch_size")),
        initial_checkpoint_name=optional_str(
            initial_checkpoint.get("preset_name")
            or initial_checkpoint.get("resolved_kind")
            or initial_checkpoint.get("mode")
        ),
        unlabeled_row_count=optional_int(manifest.get("unlabeled_row_count")),
        total_row_exposure_count=None,
        labeled_row_exposure_count=None,
        unlabeled_row_exposure_count=None,
        unique_total_row_count=None,
        unique_labeled_row_count=None,
        unique_unlabeled_row_count=None,
        train_seconds=optional_float(runtime_metrics.get("train_seconds")),
        training_example_count=optional_int(
            runtime_metrics.get("training_example_count")
        ),
        examples_per_second=optional_float(runtime_metrics.get("examples_per_second")),
        trainable_param_ratio=optional_float(
            runtime_metrics.get("trainable_param_ratio")
        ),
        peft_adapter_name=optional_str(
            peft_adapter_config.get("adapter_name")
            or peft_adapter_config.get("peft_adapter_name")
        ),
        peft_adapter_rank=optional_int(peft_adapter_config.get("rank")),
        peft_adapter_alpha=optional_int(peft_adapter_config.get("alpha")),
        peft_adapter_dropout=optional_float(peft_adapter_config.get("dropout")),
        peft_adapter_bias=optional_str(peft_adapter_config.get("bias")),
        peft_adapter_target_modules=optional_str(
            peft_adapter_config.get("target_modules")
        ),
        peft_adapter_parameters_json=json_object_snapshot(
            peft_adapter_config,
            excluded_keys={"adapter_name", "peft_adapter_name"},
        ),
        run_control_budget_name=optional_str(run_control.get("budget_name")),
        run_control_output_dir=optional_str(run_control.get("output_root")),
        client_count=None,
        round_budget=None,
        completed_rounds=None,
        shard_policy_name=None,
        shard_alpha=None,
        payload_adapter_kind=None,
        update_family_name=None,
        aggregation_backend_name=None,
        fl_composition_mode=None,
        fl_execution_role=None,
        fl_descriptor_name=None,
        update_delta_format=None,
        local_regularizer_name=None,
        local_regularizer_mu=None,
        embedding_backend=None,
        embedding_model_id=None,
        embedding_device=None,
        local_trainer_device=None,
        created_at=infer_created_at(trainer_version),
    )

    eval_metrics: list[EvalMetricRecord] = []
    per_class_metrics: list[PerClassMetricRecord] = []
    confusion_matrix_cells: list[ConfusionMatrixCellRecord] = []
    for eval_set, report in results.items():
        if not isinstance(eval_set, str) or not isinstance(report, dict):
            continue
        eval_metrics.append(
            build_eval_metric(run_id=trainer_version, eval_set=eval_set, report=report)
        )
        per_class_metrics.extend(
            build_per_class_metrics(
                run_id=trainer_version,
                eval_set=eval_set,
                per_category=as_mapping(report.get("per_category")),
            )
        )
        confusion_matrix_cells.extend(
            build_confusion_matrix_cells(
                run_id=trainer_version,
                eval_set=eval_set,
                confusion_matrix=as_mapping(report.get("confusion_matrix")),
            )
        )

    epoch_metrics: list[EpochMetricRecord] = []
    epoch_per_class_metrics: list[EpochPerClassMetricRecord] = []
    for history_record in as_sequence(manifest.get("history")):
        if not isinstance(history_record, dict):
            continue
        epoch = optional_int(history_record.get("epoch"))
        if epoch is None:
            continue
        epoch_metrics.append(
            EpochMetricRecord(
                run_id=trainer_version,
                epoch=epoch,
                train_loss=optional_float(history_record.get("train_loss")),
                train_sup_loss=optional_float(history_record.get("train_sup_loss")),
                train_unsup_loss=optional_float(history_record.get("train_unsup_loss")),
                train_util_ratio=optional_float(history_record.get("train_util_ratio")),
                selection_loss=optional_float(history_record.get("selection_loss")),
                selection_accuracy_top_1=optional_float(
                    history_record.get("selection_accuracy_top_1")
                ),
                selection_macro_f1=optional_float(
                    history_record.get("selection_macro_f1")
                ),
                selection_expected_calibration_error=optional_float(
                    history_record.get("selection_expected_calibration_error")
                ),
                selection_worst_category_f1=optional_str(
                    history_record.get("selection_worst_category_f1")
                ),
                selection_worst_category_f1_value=optional_float(
                    history_record.get("selection_worst_category_f1_value")
                ),
            )
        )
        epoch_per_class_metrics.extend(
            build_epoch_per_class_metrics(
                run_id=trainer_version,
                epoch=epoch,
                per_category=as_mapping(history_record.get("selection_per_category")),
            )
        )

    artifacts = build_projection_artifacts(
        run_id=trainer_version,
        projection_artifacts=as_mapping(manifest.get("projection_artifacts")),
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


def _peft_adapter_config_from_backbone(backbone: dict[str, Any]) -> dict[str, Any]:
    current = as_mapping(backbone.get("peft_adapter_config"))
    if current:
        parameters = as_mapping(current.get("parameters"))
        return {**parameters, **current}
    return {}


def _infer_track(*, report_path: Path, payload: dict[str, Any]) -> str:
    parts = set(report_path.parts)
    if "central_ssl_initial_eval" in parts:
        return "central_peft_initial_eval"
    if CENTRAL_PEFT_SSL_CONTROL_PATH_NAMES & parts:
        return "central_peft_ssl"
    if CENTRAL_PEFT_SUPERVISED_CONTROL_PATH_NAMES & parts:
        return "central_peft_supervised"
    if "train_classifier" in parts:
        return "central_classifier_seed"
    schema_version = str(payload.get("schema_version") or "").strip()
    return schema_version or "unknown"


def _infer_method_family(payload: dict[str, Any]) -> str:
    schema_version = str(payload.get("schema_version") or "")
    if schema_version == "central_peft_classifier_eval.v1":
        return "peft_classifier"
    return "unknown"


def _infer_method_name(
    *,
    report_path: Path,
    query_ssl_method: dict[str, Any],
) -> str:
    preset_name = optional_str(
        query_ssl_method.get("preset_name") or query_ssl_method.get("name")
    )
    if preset_name:
        return preset_name
    if report_path.name == "initial_eval.report.json":
        return "initial_eval"
    run_dir = report_path.parent.parent
    parent_name = run_dir.parent.name
    if parent_name and not parent_name.startswith("labeled-"):
        return parent_name
    return "supervised"

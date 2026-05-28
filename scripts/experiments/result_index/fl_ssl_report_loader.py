"""Load FL SSL reports into normalized result index records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.experiments.result_index.models import (
    ArtifactRecord,
    ExperimentRunRecord,
    ResultIndexRecords,
)
from scripts.experiments.result_index.record_builders import (
    build_confusion_matrix_cells,
    build_eval_metric,
    build_per_class_metrics,
    build_projection_artifacts,
)
from scripts.experiments.result_index.report_parsing import (
    as_mapping,
    infer_created_at,
    json_object_snapshot,
    optional_float,
    optional_int,
    optional_str,
)


def is_fl_ssl_report(payload: dict[str, Any]) -> bool:
    return (
        str(payload.get("track") or "") == "fl_ssl_main_comparison"
        or str(payload.get("schema_version") or "") == "federated_simulation_report.v1"
    )


def load_fl_ssl_result_index_records(
    *,
    report_path: Path,
    payload: dict[str, Any],
) -> ResultIndexRecords:
    protocol = as_mapping(payload.get("protocol"))
    metrics = as_mapping(payload.get("metrics"))
    fl_data_source = as_mapping(protocol.get("fl_data_source"))
    source_selection = as_mapping(fl_data_source.get("source_selection"))
    round_runtime = as_mapping(protocol.get("round_runtime"))
    objective = as_mapping(protocol.get("objective"))
    fl_method = as_mapping(protocol.get("fl_method"))
    manual_axes = as_mapping(fl_method.get("manual_axes"))
    ssl_method = as_mapping(protocol.get("ssl_method"))
    shard_policy = as_mapping(protocol.get("shard_policy"))
    local_update_budget = as_mapping(protocol.get("local_update_budget"))
    embedding_adapter = as_mapping(protocol.get("embedding_adapter"))
    local_trainer_runtime = as_mapping(protocol.get("local_trainer_runtime"))
    run_control = as_mapping(protocol.get("run_control"))
    split_summary = as_mapping(protocol.get("labeled_unlabeled_split"))

    run_id = infer_fl_ssl_run_id(report_path)
    method_name = (
        optional_str(objective.get("query_ssl.method_name"))
        or optional_str(ssl_method.get("name"))
        or "unknown"
    )
    payload_adapter_kind = optional_str(round_runtime.get("payload_adapter_kind"))
    update_family = optional_str(
        round_runtime.get("update_family_name")
    ) or optional_str(manual_axes.get("update_family"))
    composition_mode = optional_str(fl_method.get("composition_mode"))
    descriptor_name = optional_str(fl_method.get("descriptor_name")) or optional_str(
        ssl_method.get("name")
    )
    local_regularizer_name, local_regularizer_mu = infer_local_regularizer(objective)
    run = ExperimentRunRecord(
        run_id=run_id,
        track=optional_str(payload.get("track")) or "fl_ssl_main_comparison",
        method_family=infer_fl_method_family(
            composition_mode=composition_mode,
            descriptor_name=descriptor_name,
            update_family=update_family,
        ),
        method_name=method_name,
        algorithm_name=optional_str(objective.get("query_ssl.algorithm_name")),
        selection_slug=optional_str(fl_data_source.get("split_id")),
        labeled_dataset_name=optional_str(source_selection.get("labeled")),
        unlabeled_dataset_name=optional_str(source_selection.get("unlabeled")),
        validation_dataset_name=optional_str(source_selection.get("validation")),
        test_dataset_name=optional_str(source_selection.get("test")),
        seed=optional_int(protocol.get("seed")),
        learning_rate=optional_float(local_update_budget.get("learning_rate")),
        classifier_learning_rate=None,
        epochs=optional_int(local_update_budget.get("local_epochs")),
        max_train_steps=optional_int(local_update_budget.get("max_steps")),
        train_batch_size=optional_int(local_update_budget.get("batch_size")),
        eval_batch_size=None,
        initial_checkpoint_name=None,
        unlabeled_row_count=optional_int(split_summary.get("actual_unlabeled_count")),
        total_row_exposure_count=optional_int(
            split_summary.get("actual_total_exposure_count")
        ),
        labeled_row_exposure_count=optional_int(
            split_summary.get("actual_labeled_exposure_count")
            or split_summary.get("actual_labeled_count")
        ),
        unlabeled_row_exposure_count=optional_int(
            split_summary.get("actual_unlabeled_exposure_count")
            or split_summary.get("actual_unlabeled_count")
        ),
        unique_total_row_count=optional_int(split_summary.get("unique_total_count")),
        unique_labeled_row_count=optional_int(
            split_summary.get("unique_labeled_count")
        ),
        unique_unlabeled_row_count=optional_int(
            split_summary.get("unique_unlabeled_count")
        ),
        train_seconds=None,
        training_example_count=None,
        examples_per_second=None,
        trainable_param_ratio=None,
        peft_adapter_name=optional_str(
            _trainable_state_objective_value(objective, "peft_adapter_name")
        ),
        peft_adapter_rank=optional_int(
            _trainable_state_objective_value(objective, "rank")
        ),
        peft_adapter_alpha=optional_int(
            _trainable_state_objective_value(objective, "alpha")
        ),
        peft_adapter_dropout=optional_float(
            _trainable_state_objective_value(objective, "dropout")
        ),
        peft_adapter_bias=optional_str(
            _trainable_state_objective_value(objective, "bias")
        ),
        peft_adapter_target_modules=optional_str(
            _trainable_state_objective_value(objective, "target_modules")
        ),
        peft_adapter_parameters_json=json_object_snapshot(
            _peft_adapter_objective_parameters(
                objective=objective,
                payload_adapter_kind=payload_adapter_kind,
            ),
            excluded_keys={
                "delta_format",
                "peft_adapter_name",
                "proximal_mu",
            },
        ),
        run_control_budget_name=optional_str(run_control.get("budget_name")),
        run_control_output_dir=optional_str(run_control.get("output_dir")),
        client_count=optional_int(protocol.get("client_count")),
        round_budget=optional_int(protocol.get("round_budget")),
        completed_rounds=optional_int(protocol.get("completed_rounds")),
        shard_policy_name=optional_str(shard_policy.get("name")),
        shard_alpha=optional_float(shard_policy.get("alpha")),
        payload_adapter_kind=payload_adapter_kind,
        update_family_name=update_family,
        aggregation_backend_name=optional_str(
            round_runtime.get("aggregation_backend_name")
        ),
        fl_composition_mode=composition_mode,
        fl_execution_role=optional_str(fl_method.get("execution_role")),
        fl_descriptor_name=descriptor_name,
        update_delta_format=optional_str(
            _trainable_state_objective_value(objective, "delta_format")
        ),
        local_regularizer_name=local_regularizer_name,
        local_regularizer_mu=local_regularizer_mu,
        embedding_backend=optional_str(embedding_adapter.get("backend")),
        embedding_model_id=optional_str(embedding_adapter.get("model_id")),
        embedding_device=optional_str(embedding_adapter.get("device")),
        local_trainer_device=optional_str(local_trainer_runtime.get("device")),
        created_at=infer_created_at(run_id),
    )

    eval_metrics = []
    per_class_metrics = []
    confusion_matrix_cells = []
    for eval_set in ("initial_validation", "final_validation"):
        report = as_mapping(metrics.get(eval_set))
        if not report:
            continue
        eval_metrics.append(
            build_eval_metric(run_id=run_id, eval_set=eval_set, report=report)
        )
        per_class_metrics.extend(
            build_per_class_metrics(
                run_id=run_id,
                eval_set=eval_set,
                per_category=as_mapping(report.get("per_category")),
            )
        )
        confusion_matrix_cells.extend(
            build_confusion_matrix_cells(
                run_id=run_id,
                eval_set=eval_set,
                confusion_matrix=as_mapping(report.get("confusion_matrix")),
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
    split_manifest_path = optional_str(fl_data_source.get("split_manifest_path"))
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
    artifacts.extend(
        build_projection_artifacts(
            run_id=run_id,
            projection_artifacts=_load_fl_ssl_projection_artifacts(
                report_path=report_path,
                payload=payload,
            ),
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


def _load_fl_ssl_projection_artifacts(
    *,
    report_path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = as_mapping(payload.get("diagnostics"))
    projection_artifacts = as_mapping(diagnostics.get("final_projection_artifacts"))
    if projection_artifacts and _projection_artifacts_have_existing_figures(
        projection_artifacts
    ):
        return projection_artifacts

    manifest_path = (
        report_path.parent.parent / "projections" / "projection_manifest.json"
    )
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "enabled": True,
        "manifest_path": str(manifest_path),
        "datasets": as_mapping(manifest.get("datasets")),
    }


def _projection_artifacts_have_existing_figures(
    projection_artifacts: dict[str, Any],
) -> bool:
    datasets = as_mapping(projection_artifacts.get("datasets"))
    figure_paths = [
        entry.get("figure_png")
        for entry in datasets.values()
        if isinstance(entry, dict) and entry.get("figure_png")
    ]
    return bool(figure_paths) and all(Path(str(path)).exists() for path in figure_paths)


def infer_fl_method_family(
    *,
    composition_mode: str | None,
    descriptor_name: str | None,
    update_family: str | None,
) -> str:
    if str(composition_mode or "").strip().lower() == "manual":
        return "manual_baselines"
    return descriptor_name or update_family or "unknown"


def infer_local_regularizer(objective: dict[str, Any]) -> tuple[str, float | None]:
    proximal_mu = optional_float(
        _trainable_state_objective_value(objective, "proximal_mu")
    )
    if proximal_mu is not None and proximal_mu > 0:
        return "fedprox", proximal_mu
    return "none", None


def _trainable_state_objective_value(objective: dict[str, Any], key: str) -> object:
    """objective payload에서 update-family scoped 값을 generic하게 찾는다."""

    direct_value = objective.get(key)
    if direct_value is not None:
        return direct_value
    dotted_suffix = f".{key}"
    values: list[object] = []
    for raw_key, value in objective.items():
        if str(raw_key).endswith(dotted_suffix):
            values.append(value)
        if isinstance(value, dict):
            nested_value = _trainable_state_objective_value(value, key)
            if nested_value is not None:
                values.append(nested_value)
    normalized_values = {str(value) for value in values if value is not None}
    if len(normalized_values) > 1:
        raise ValueError(
            f"FL SSL objective has conflicting {key!r} values: "
            f"{sorted(normalized_values)}"
        )
    if values:
        return values[0]
    return None


def _peft_adapter_objective_parameters(
    *,
    objective: dict[str, Any],
    payload_adapter_kind: str | None,
) -> dict[str, Any]:
    if payload_adapter_kind is None:
        return {}
    parameters: dict[str, Any] = {}
    for raw_key, value in objective.items():
        key = str(raw_key)
        prefix = f"{payload_adapter_kind}."
        if key.startswith(prefix):
            parameters[key.removeprefix(prefix)] = value
            continue
        if key == payload_adapter_kind and isinstance(value, dict):
            parameters.update(value)
    return parameters


def infer_fl_ssl_run_id(report_path: Path) -> str:
    layout_parts = _fl_ssl_layout_parts(report_path)
    if layout_parts:
        return "__".join(layout_parts)

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


def _fl_ssl_layout_parts(report_path: Path) -> tuple[str, ...]:
    parts = report_path.parts
    if "fl_ssl" not in parts:
        return ()
    fl_ssl_index = parts.index("fl_ssl")
    layout_parts = parts[fl_ssl_index + 1 :]
    if len(layout_parts) < 3 or layout_parts[-2] != "reports":
        return ()
    return tuple(part for part in layout_parts[:-2] if part)

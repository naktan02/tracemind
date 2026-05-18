"""Build FL SSL dashboard view-models from indexed report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_fl_ssl_dashboard_views(
    *,
    runs: list[dict[str, Any]],
    eval_metrics: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "fl_ssl_runs": _build_fl_ssl_runs(
            runs=runs,
            eval_metrics=eval_metrics,
            artifacts=artifacts,
        ),
        "fl_ssl_rounds": _build_fl_ssl_rounds(runs=runs, artifacts=artifacts),
        "fl_ssl_client_rounds": _build_fl_ssl_client_rounds(
            runs=runs,
            artifacts=artifacts,
        ),
        "fl_ssl_client_validations": _build_fl_ssl_client_validations(
            runs=runs,
            artifacts=artifacts,
        ),
        "fl_ssl_client_splits": _build_fl_ssl_client_splits(
            runs=runs,
            artifacts=artifacts,
        ),
    }


def _build_fl_ssl_runs(
    *,
    runs: list[dict[str, Any]],
    eval_metrics: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    eval_by_run = _group_eval_metrics(eval_metrics)
    rows: list[dict[str, Any]] = []
    for run, report_artifact, report_payload in _fl_ssl_report_contexts(
        runs=runs,
        artifacts=artifacts,
    ):
        run_id = str(run["run_id"])
        metrics = _as_mapping(report_payload.get("metrics"))
        primary = _as_mapping(metrics.get("primary"))
        secondary = _as_mapping(metrics.get("secondary"))
        client_validation = _as_mapping(metrics.get("client_validation"))
        diagnostics = _as_mapping(report_payload.get("diagnostics"))
        run_eval_metrics = eval_by_run.get(run_id, {})
        initial_validation = run_eval_metrics.get("initial_validation", {})
        final_validation = run_eval_metrics.get("final_validation", {})

        rows.append(
            {
                **run,
                "report_path": (
                    str(report_artifact["artifact_ref"])
                    if report_artifact is not None
                    else None
                ),
                "initial_macro_f1": initial_validation.get("macro_f1"),
                "initial_loss": initial_validation.get("loss"),
                "initial_expected_calibration_error": initial_validation.get(
                    "expected_calibration_error"
                ),
                "final_macro_f1": final_validation.get("macro_f1"),
                "final_loss": final_validation.get("loss"),
                "final_accuracy_top_1": final_validation.get("accuracy_top_1"),
                "final_expected_calibration_error": final_validation.get(
                    "expected_calibration_error"
                ),
                "macro_f1": _first_present(
                    primary.get("macro_f1"),
                    final_validation.get("macro_f1"),
                ),
                "loss": _first_present(
                    secondary.get("loss"),
                    final_validation.get("loss"),
                ),
                "expected_calibration_error": _first_present(
                    secondary.get("expected_calibration_error"),
                    final_validation.get("expected_calibration_error"),
                ),
                "worst_client_macro_f1": _first_present(
                    primary.get("worst_client_macro_f1"),
                    client_validation.get("worst_client_macro_f1"),
                ),
                "best_client_macro_f1": client_validation.get("best_client_macro_f1"),
                "macro_f1_std": client_validation.get("macro_f1_std"),
                "loss_std": client_validation.get("loss_std"),
                "fairness_gap": client_validation.get("fairness_gap"),
                "evaluated_client_count": client_validation.get(
                    "evaluated_client_count"
                ),
                "per_client_macro_f1_variance": secondary.get(
                    "per_client_macro_f1_variance"
                ),
                "communication_cost": _first_present(
                    secondary.get("communication_cost"),
                    diagnostics.get("communication_cost"),
                ),
            }
        )
    return rows


def _build_fl_ssl_rounds(
    *,
    runs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run, _report_artifact, report_payload in _fl_ssl_report_contexts(
        runs=runs,
        artifacts=artifacts,
    ):
        run_id = str(run["run_id"])
        rounds = [
            round_record
            for round_record in _as_sequence(report_payload.get("rounds"))
            if isinstance(round_record, dict)
        ]
        rounds_by_id = {
            str(round_record.get("round_id")): round_record
            for round_record in rounds
            if round_record.get("round_id") is not None
        }
        metrics = _as_mapping(report_payload.get("metrics"))
        round_progression = _as_mapping(metrics.get("round_progression"))
        validation_curve = [
            point
            for point in _as_sequence(round_progression.get("validation_curve"))
            if isinstance(point, dict)
        ]
        if not validation_curve:
            validation_curve = [
                {
                    **_as_mapping(
                        round_record.get("global_validation")
                        or round_record.get("validation")
                    ),
                    "round_id": round_record.get("round_id"),
                    "round_index": round_record.get("round_index"),
                }
                for round_record in rounds
            ]

        for point in validation_curve:
            round_id = str(point.get("round_id") or "")
            round_record = rounds_by_id.get(round_id, {})
            validation = _as_mapping(
                round_record.get("global_validation") or round_record.get("validation")
            )
            delta_from_initial = _as_mapping(round_record.get("delta_from_initial"))
            delta_from_previous = _as_mapping(
                round_record.get("delta_from_previous_round")
            )
            rows.append(
                {
                    "run_id": run_id,
                    "method_name": run.get("method_name"),
                    "algorithm_name": run.get("algorithm_name"),
                    "client_count": run.get("client_count"),
                    "round_budget": run.get("round_budget"),
                    "completed_rounds": run.get("completed_rounds"),
                    "round_id": round_id or None,
                    "round_index": _first_present(
                        point.get("round_index"),
                        round_record.get("round_index"),
                    ),
                    "macro_f1": _first_present(
                        point.get("macro_f1"),
                        validation.get("macro_f1"),
                    ),
                    "accuracy_top_1": _first_present(
                        point.get("accuracy_top_1"),
                        validation.get("accuracy_top_1"),
                    ),
                    "loss": _first_present(point.get("loss"), validation.get("loss")),
                    "expected_calibration_error": _first_present(
                        point.get("expected_calibration_error"),
                        validation.get("expected_calibration_error"),
                    ),
                    "accepted_ratio": _first_present(
                        point.get("accepted_ratio"),
                        validation.get("accepted_ratio"),
                    ),
                    "update_count": round_record.get("update_count"),
                    "total_payload_bytes": round_record.get("total_payload_bytes"),
                    "round_time_seconds": round_record.get("round_time_seconds"),
                    "gpu_memory_peak_mb": round_record.get("gpu_memory_peak_mb"),
                    "loss_delta_from_initial": delta_from_initial.get("loss_delta"),
                    "macro_f1_delta_from_initial": delta_from_initial.get(
                        "macro_f1_delta"
                    ),
                    "accuracy_delta_from_initial": delta_from_initial.get(
                        "accuracy_top_1_delta"
                    ),
                    "ece_delta_from_initial": delta_from_initial.get(
                        "expected_calibration_error_delta"
                    ),
                    "accepted_ratio_delta_from_initial": delta_from_initial.get(
                        "accepted_ratio_delta"
                    ),
                    "loss_delta_from_previous": delta_from_previous.get("loss_delta"),
                    "macro_f1_delta_from_previous": delta_from_previous.get(
                        "macro_f1_delta"
                    ),
                    "accuracy_delta_from_previous": delta_from_previous.get(
                        "accuracy_top_1_delta"
                    ),
                    "ece_delta_from_previous": delta_from_previous.get(
                        "expected_calibration_error_delta"
                    ),
                    "accepted_ratio_delta_from_previous": delta_from_previous.get(
                        "accepted_ratio_delta"
                    ),
                }
            )
    return rows


def _build_fl_ssl_client_rounds(
    *,
    runs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run, _report_artifact, report_payload in _fl_ssl_report_contexts(
        runs=runs,
        artifacts=artifacts,
    ):
        run_id = str(run["run_id"])
        for round_record in _as_sequence(report_payload.get("rounds")):
            round_mapping = _as_mapping(round_record)
            if not round_mapping:
                continue
            round_id = round_mapping.get("round_id")
            round_index = round_mapping.get("round_index")
            for client_record in _as_sequence(round_mapping.get("clients")):
                client = _as_mapping(client_record)
                if not client:
                    continue
                rows.append(
                    {
                        "run_id": run_id,
                        "method_name": run.get("method_name"),
                        "algorithm_name": run.get("algorithm_name"),
                        "client_id": client.get("client_id"),
                        "round_id": round_id,
                        "round_index": round_index,
                        "candidate_count": client.get("candidate_count"),
                        "accepted_count": client.get("accepted_count"),
                        "accepted_ratio": client.get("accepted_ratio"),
                        "update_generated": client.get("update_generated"),
                        "aggregation_example_count": client.get(
                            "aggregation_example_count"
                        ),
                        "delta_l2_norm": client.get("delta_l2_norm"),
                        "client_payload_bytes": client.get("client_payload_bytes"),
                        "client_train_time_seconds": client.get(
                            "client_train_time_seconds"
                        ),
                        "candidate_confidence_mean": client.get(
                            "candidate_confidence_mean"
                        ),
                        "candidate_margin_mean": client.get("candidate_margin_mean"),
                        "pseudo_label_confidence_mean": client.get(
                            "pseudo_label_confidence_mean"
                        ),
                        "pseudo_label_margin_mean": client.get(
                            "pseudo_label_margin_mean"
                        ),
                        "pseudo_label_accuracy": client.get("pseudo_label_accuracy"),
                        "pseudo_label_correct_count": client.get(
                            "pseudo_label_correct_count"
                        ),
                        "pseudo_label_evaluated_count": client.get(
                            "pseudo_label_evaluated_count"
                        ),
                        "accepted_label_distribution": client.get(
                            "accepted_label_distribution"
                        ),
                        "rejected_label_distribution": client.get(
                            "rejected_label_distribution"
                        ),
                    }
                )
    return rows


def _build_fl_ssl_client_validations(
    *,
    runs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run, _report_artifact, report_payload in _fl_ssl_report_contexts(
        runs=runs,
        artifacts=artifacts,
    ):
        run_id = str(run["run_id"])
        metrics = _as_mapping(report_payload.get("metrics"))
        client_validation = _as_mapping(metrics.get("client_validation"))
        for client_record in _as_sequence(client_validation.get("clients")):
            client = _as_mapping(client_record)
            if not client:
                continue
            validation = _as_mapping(client.get("validation"))
            rows.append(
                {
                    "run_id": run_id,
                    "method_name": run.get("method_name"),
                    "algorithm_name": run.get("algorithm_name"),
                    "client_id": client.get("client_id"),
                    "latest_round_id": client.get("latest_round_id"),
                    "client_train_size": client.get("client_train_size"),
                    "client_labeled_count": client.get("client_labeled_count"),
                    "client_unlabeled_count": client.get("client_unlabeled_count"),
                    "client_label_distribution": client.get(
                        "client_label_distribution"
                    ),
                    "client_candidate_count": client.get("client_candidate_count"),
                    "client_accepted_count": client.get("client_accepted_count"),
                    "client_accepted_ratio": client.get("client_accepted_ratio"),
                    "aggregation_example_count": client.get(
                        "aggregation_example_count"
                    ),
                    "client_payload_bytes": client.get("client_payload_bytes"),
                    "client_update_generated": client.get("client_update_generated"),
                    "latest_update_generated": client.get("latest_update_generated"),
                    "update_generated_round_count": client.get(
                        "update_generated_round_count"
                    ),
                    "client_delta_l2_norm": client.get("client_delta_l2_norm"),
                    "mean_delta_l2_norm": client.get("mean_delta_l2_norm"),
                    "max_delta_l2_norm": client.get("max_delta_l2_norm"),
                    "update_norm_variance": client.get("update_norm_variance"),
                    "client_train_time_seconds": client.get(
                        "client_train_time_seconds"
                    ),
                    "mean_client_train_time_seconds": client.get(
                        "mean_client_train_time_seconds"
                    ),
                    "pseudo_label_accuracy": client.get("pseudo_label_accuracy"),
                    "client_validation_loss": _first_present(
                        client.get("client_validation_loss"),
                        validation.get("loss"),
                    ),
                    "client_validation_macro_f1": _first_present(
                        client.get("client_validation_macro_f1"),
                        validation.get("macro_f1"),
                    ),
                    "client_validation_accuracy_top_1": validation.get(
                        "accuracy_top_1"
                    ),
                    "client_validation_ece": _first_present(
                        client.get("client_validation_ece"),
                        validation.get("expected_calibration_error"),
                    ),
                    "client_validation_rows_total": validation.get("rows_total"),
                }
            )
    return rows


def _build_fl_ssl_client_splits(
    *,
    runs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    split_artifacts = {
        str(artifact["run_id"]): artifact
        for artifact in artifacts
        if artifact.get("artifact_kind") == "fl_client_split_manifest"
    }
    rows: list[dict[str, Any]] = []
    for run, _report_artifact, report_payload in _fl_ssl_report_contexts(
        runs=runs,
        artifacts=artifacts,
    ):
        run_id = str(run["run_id"])
        split_artifact = split_artifacts.get(run_id)
        split_payload = _read_json_object(
            Path(str(split_artifact["artifact_ref"]))
            if split_artifact is not None
            else None
        )
        split_clients = [
            client
            for client in _as_sequence(split_payload.get("clients"))
            if isinstance(client, dict)
        ]
        if split_clients:
            for client in split_clients:
                rows.append(
                    {
                        "run_id": run_id,
                        "method_name": run.get("method_name"),
                        "algorithm_name": run.get("algorithm_name"),
                        "client_id": client.get("client_id"),
                        "source": "split_manifest",
                        "labeled_count": client.get("labeled_count"),
                        "unlabeled_count": client.get("unlabeled_count"),
                        "labeled_label_distribution": client.get(
                            "labeled_label_distribution"
                        ),
                        "unlabeled_label_distribution": client.get(
                            "unlabeled_label_distribution"
                        ),
                        "label_distribution": None,
                    }
                )
            continue

        metrics = _as_mapping(report_payload.get("metrics"))
        client_validation = _as_mapping(metrics.get("client_validation"))
        for client_record in _as_sequence(client_validation.get("clients")):
            client = _as_mapping(client_record)
            if not client:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "method_name": run.get("method_name"),
                    "algorithm_name": run.get("algorithm_name"),
                    "client_id": client.get("client_id"),
                    "source": "client_validation",
                    "labeled_count": client.get("client_labeled_count"),
                    "unlabeled_count": client.get("client_unlabeled_count"),
                    "labeled_label_distribution": None,
                    "unlabeled_label_distribution": None,
                    "label_distribution": client.get("client_label_distribution"),
                }
            )
    return rows


def _fl_ssl_report_contexts(
    *,
    runs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]]:
    report_artifacts = {
        str(artifact["run_id"]): artifact
        for artifact in artifacts
        if artifact.get("artifact_kind") == "fl_ssl_report"
    }
    contexts: list[tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]] = []
    for run in runs:
        if not _is_fl_ssl_track(run.get("track")):
            continue
        run_id = str(run["run_id"])
        report_artifact = report_artifacts.get(run_id)
        report_payload = _read_json_object(
            Path(str(report_artifact["artifact_ref"]))
            if report_artifact is not None
            else None
        )
        contexts.append((run, report_artifact, report_payload))
    return contexts


def _group_eval_metrics(
    eval_metrics: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for metric in eval_metrics:
        run_id = str(metric.get("run_id"))
        eval_set = str(metric.get("eval_set"))
        grouped.setdefault(run_id, {})[eval_set] = metric
    return grouped


def _read_json_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _as_mapping(payload)


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _is_fl_ssl_track(track: Any) -> bool:
    return str(track or "").startswith("fl_ssl")

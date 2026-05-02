"""experiment run stdout와 artifact에서 결과 요약을 추출한다."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from main_server.src.services.experiment_workspace.payloads import (
    ExperimentRunMetricPayload,
    ExperimentRunResultSummaryPayload,
)

_REPORTED_OUTPUT_PATTERN = re.compile(r"^(?P<key>[A-Za-z0-9_]+)=(?P<value>.+)$")
_NUMERIC_REPORT_KEYS = (
    "rows_total",
    "loss",
    "accuracy_top_1",
    "mean_true_label_probability",
    "mean_top_1_probability",
    "mean_margin_top1_top2",
)
_TEACHER_SUMMARY_KEYS = (
    "accepted_count",
    "accepted_ratio",
    "accepted_hidden_label_accuracy",
)


def extract_reported_outputs(
    stdout_log_path: Path,
) -> dict[str, str]:
    """stdout의 `key=value` 줄에서 산출물 경로/메타데이터를 추출한다."""

    if not stdout_log_path.exists():
        return {}

    outputs: dict[str, str] = {}
    for raw_line in stdout_log_path.read_text(encoding="utf-8").splitlines():
        match = _REPORTED_OUTPUT_PATTERN.match(raw_line.strip())
        if match is None:
            continue
        outputs[match.group("key")] = match.group("value").strip()
    return outputs


def build_experiment_run_result_summary(
    *,
    reported_outputs: Mapping[str, str],
    repo_root: Path,
) -> ExperimentRunResultSummaryPayload | None:
    """report/summary JSON에서 비교 가능한 metric 집합을 만든다."""

    metric_map: dict[str, float] = {}
    source_paths: list[str] = []

    report_path = _resolve_output_path(reported_outputs.get("report_json"), repo_root)
    if report_path is not None and report_path.exists():
        metric_map.update(_extract_report_metrics(report_path))
        source_paths.append(str(report_path))

    prediction_summary_path = _resolve_output_path(
        reported_outputs.get("prediction_summary_json"),
        repo_root,
    )
    if prediction_summary_path is not None and prediction_summary_path.exists():
        metric_map.update(_extract_teacher_summary_metrics(prediction_summary_path))
        source_paths.append(str(prediction_summary_path))

    if not metric_map:
        return None

    if len(source_paths) > 1:
        source_kind = "combined_outputs"
    elif report_path is not None and report_path.exists():
        source_kind = "report_json"
    else:
        source_kind = "prediction_summary_json"

    return ExperimentRunResultSummaryPayload(
        source_kind=source_kind,
        source_paths=tuple(source_paths),
        metrics=tuple(
            ExperimentRunMetricPayload(metric_key=metric_key, value=value)
            for metric_key, value in sorted(metric_map.items())
        ),
    )


def _resolve_output_path(raw_path: str | None, repo_root: Path) -> Path | None:
    if raw_path is None or not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def _extract_report_metrics(report_path: Path) -> dict[str, float]:
    payload = _load_json_object(report_path)
    manifest = payload.get("manifest")
    results = payload.get("results")

    metrics: dict[str, float] = {}
    if isinstance(manifest, Mapping):
        best_selection_report = manifest.get("best_selection_report")
        if isinstance(best_selection_report, Mapping):
            metrics.update(
                _flatten_eval_report("selection", best_selection_report)
            )

    if isinstance(results, Mapping):
        for dataset_name, report in sorted(results.items()):
            if isinstance(report, Mapping):
                metrics.update(_flatten_eval_report(str(dataset_name), report))
    return metrics


def _extract_teacher_summary_metrics(summary_path: Path) -> dict[str, float]:
    payload = _load_json_object(summary_path)
    metrics: dict[str, float] = {}
    for metric_key in _TEACHER_SUMMARY_KEYS:
        numeric_value = _coerce_numeric(payload.get(metric_key))
        if numeric_value is None:
            continue
        metrics[f"teacher.{metric_key}"] = numeric_value
    return metrics


def _flatten_eval_report(
    prefix: str,
    report: Mapping[str, Any],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric_key in _NUMERIC_REPORT_KEYS:
        numeric_value = _coerce_numeric(report.get(metric_key))
        if numeric_value is None:
            continue
        metrics[f"{prefix}.{metric_key}"] = numeric_value

    per_category = report.get("per_category")
    if isinstance(per_category, Mapping):
        metrics.update(_summarize_per_category_metrics(prefix, per_category))
    return metrics


def _summarize_per_category_metrics(
    prefix: str,
    per_category: Mapping[str, Any],
) -> dict[str, float]:
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[float] = []

    for category_metrics in per_category.values():
        if not isinstance(category_metrics, Mapping):
            continue
        precision = _coerce_numeric(category_metrics.get("precision"))
        recall = _coerce_numeric(category_metrics.get("recall"))
        f1 = _coerce_numeric(category_metrics.get("f1"))
        support = _coerce_numeric(category_metrics.get("support"))
        if precision is None or recall is None or f1 is None or support is None:
            continue
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    if not f1s:
        return {}

    total_support = sum(supports)
    weighted_f1 = (
        sum(metric * support for metric, support in zip(f1s, supports, strict=True))
        / total_support
        if total_support > 0
        else 0.0
    )
    weighted_precision = (
        sum(
            metric * support
            for metric, support in zip(precisions, supports, strict=True)
        )
        / total_support
        if total_support > 0
        else 0.0
    )
    weighted_recall = (
        sum(metric * support for metric, support in zip(recalls, supports, strict=True))
        / total_support
        if total_support > 0
        else 0.0
    )
    return {
        f"{prefix}.macro_precision": sum(precisions) / len(precisions),
        f"{prefix}.macro_recall": sum(recalls) / len(recalls),
        f"{prefix}.macro_f1": sum(f1s) / len(f1s),
        f"{prefix}.weighted_precision": weighted_precision,
        f"{prefix}.weighted_recall": weighted_recall,
        f"{prefix}.weighted_f1": weighted_f1,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object payload: {path}")
    return payload


def _coerce_numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None

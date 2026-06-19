"""중앙 SSL report 조립에서 사용하는 결과/selection report 보조 유틸."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_SELECTION_REPORT_FIELD_MAP = (
    ("loss", "selection_loss"),
    ("accuracy_top_1", "selection_accuracy_top_1"),
    ("macro_f1", "selection_macro_f1"),
    ("expected_calibration_error", "selection_expected_calibration_error"),
    ("worst_category_f1", "selection_worst_category_f1"),
    ("worst_category_f1_value", "selection_worst_category_f1_value"),
)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def extract_selection_report_from_history_record(
    record: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """history epoch record에서 selection report를 추출한다."""

    if record is None:
        return None
    explicit_report = _as_mapping(record.get("selection_report"))
    if explicit_report is not None:
        return dict(explicit_report)

    if not all(
        required in record
        for required in ("selection_loss", "selection_accuracy_top_1")
    ):
        return None

    reconstructed: dict[str, Any] = {
        "loss": record["selection_loss"],
        "accuracy_top_1": record["selection_accuracy_top_1"],
    }
    for target_key, source_key in _SELECTION_REPORT_FIELD_MAP:
        if source_key in record:
            reconstructed[target_key] = record[source_key]
    per_category = _as_mapping(record.get("selection_per_category"))
    if per_category is not None:
        reconstructed["per_category"] = dict(per_category)
    return reconstructed


def extract_final_selection_report(
    history: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    """history 마지막 epoch 기준 최종 selection report를 반환한다."""

    if not history:
        return None
    return extract_selection_report_from_history_record(history[-1])


def _build_final_report(
    final_selection_report: Mapping[str, Any] | None,
    selection_set_report: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if final_selection_report is None and selection_set_report is None:
        return None
    base_report = (
        dict(final_selection_report) if final_selection_report is not None else {}
    )
    if selection_set_report is None:
        return base_report
    if "per_category" not in base_report and "per_category" in selection_set_report:
        base_report["per_category"] = dict(selection_set_report["per_category"])
    if (
        "confusion_matrix" not in base_report
        and "confusion_matrix" in selection_set_report
    ):
        base_report["confusion_matrix"] = dict(selection_set_report["confusion_matrix"])
    return base_report


def merge_results_with_best_and_final(
    *,
    results: Mapping[str, Any],
    selection_set: str,
    final_selection_report: Mapping[str, Any] | None = None,
    include_selection_set_result: bool = True,
) -> dict[str, Any]:
    """results에 best/final alias를 보강해 report 하위 호환성을 유지한다."""

    merged: dict[str, Any] = {
        str(key): dict(value) if isinstance(value, Mapping) else value
        for key, value in results.items()
    }
    selection_key = str(selection_set)
    best_report = _as_mapping(merged.get(selection_key))
    if best_report is None:
        fallback_keys = ("test", "validation")
        for fallback_key in fallback_keys:
            candidate = _as_mapping(merged.get(fallback_key))
            if candidate is not None:
                best_report = candidate
                break
    if best_report is None:
        for key, value in merged.items():
            if key in {"best", "final"}:
                continue
            candidate = _as_mapping(value)
            if candidate is not None:
                best_report = candidate
                break
    selection_set_report = _as_mapping(merged.get(selection_key))
    if best_report is not None:
        merged.setdefault("best", dict(best_report))
    if final_selection_report is not None:
        final_report = _build_final_report(
            final_selection_report=final_selection_report,
            selection_set_report=selection_set_report,
        )
        if final_report is not None:
            merged.setdefault("final", dict(final_report))
        if not include_selection_set_result and selection_key not in {"best", "final"}:
            merged.pop(selection_key, None)
        return merged
    if selection_set_report is not None:
        merged.setdefault("final", dict(selection_set_report))
    if not include_selection_set_result and selection_key not in {"best", "final"}:
        merged.pop(selection_key, None)
    return merged

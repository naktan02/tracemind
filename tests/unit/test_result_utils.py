"""query_ssl result 유틸 테스트."""

from __future__ import annotations

from scripts.support.query_ssl_text_encoder.result_utils import (
    extract_final_selection_report,
    merge_results_with_best_and_final,
)


def test_merge_results_final_adds_detail_from_selection_set() -> None:
    results = {
        "validation": {
            "accuracy_top_1": 0.91,
            "macro_f1": 0.81,
            "per_category": {
                "cat_a": {"f1": 0.9},
            },
            "confusion_matrix": {
                "cat_a": {"cat_a": 10},
            },
        }
    }

    merged = merge_results_with_best_and_final(
        results=results,
        selection_set="validation",
        final_selection_report={"macro_f1": 0.77},
    )

    assert merged["best"] == results["validation"]
    assert merged["final"]["macro_f1"] == 0.77
    assert (
        merged["final"]["per_category"] == results["validation"]["per_category"]
    )
    assert (
        merged["final"]["confusion_matrix"]
        == results["validation"]["confusion_matrix"]
    )


def test_merge_results_final_falls_back_when_selection_set_missing() -> None:
    results = {
        "validation": {
            "accuracy_top_1": 0.88,
            "macro_f1": 0.79,
        }
    }
    final_selection_report = {
        "macro_f1": 0.75,
        "accuracy_top_1": 0.87,
        "per_category": {"cat_a": {"f1": 0.5}},
    }

    merged = merge_results_with_best_and_final(
        results=results,
        selection_set="test",
        final_selection_report=final_selection_report,
    )

    assert merged["best"] == results["validation"]
    assert merged["final"] == final_selection_report


def test_extract_final_selection_report_maps_selection_fields() -> None:
    history = [
        {
            "selection_loss": 1.0,
            "selection_accuracy_top_1": 0.5,
            "selection_macro_f1": 0.6,
        }
    ]
    report = extract_final_selection_report(history)
    assert report == {
        "loss": 1.0,
        "accuracy_top_1": 0.5,
        "macro_f1": 0.6,
    }

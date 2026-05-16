"""classification_report 공통 유틸리티 tests."""

from __future__ import annotations

import pytest

from methods.evaluation.classification_report import (
    build_classification_evaluation_report,
)
from shared.src.domain.services.classification_report import (
    build_confusion_matrix,
    render_confusion_table,
    render_per_category_table,
    summarize_per_category,
)


def test_build_confusion_matrix_counts_actual_and_predicted_pairs() -> None:
    matrix = build_confusion_matrix(
        categories=["a", "b"],
        actual_labels=["a", "a", "b"],
        predicted_labels=["a", "b", "b"],
    )

    assert matrix == {
        "a": {"a": 1, "b": 1},
        "b": {"a": 0, "b": 1},
    }


def test_summarize_per_category_supports_custom_metric_keys() -> None:
    summary = summarize_per_category(
        categories=["a", "b"],
        actual_labels=["a", "a", "b"],
        predicted_labels=["a", "b", "b"],
        primary_values=[0.9, 0.2, 0.8],
        top_1_values=[0.9, 0.7, 0.8],
        margins=[0.4, 0.1, 0.3],
        primary_metric_key="mean_true_label_probability",
        top_1_metric_key="mean_top_1_probability",
    )

    assert summary["a"]["support"] == 2
    assert summary["a"]["correct"] == 1
    assert summary["a"]["mean_true_label_probability"] == 0.55
    assert summary["a"]["mean_top_1_probability"] == 0.8
    assert summary["b"]["precision"] == 0.5


def test_summarize_per_category_can_skip_rounding() -> None:
    summary = summarize_per_category(
        categories=["a"],
        actual_labels=["a", "a", "a"],
        predicted_labels=["a", "a", "a"],
        primary_values=[0.1111114, 0.1111115, 0.1111116],
        top_1_values=[0.2222224, 0.2222225, 0.2222226],
        margins=[0.3333334, 0.3333335, 0.3333336],
        primary_metric_key="mean_true_label_score",
        top_1_metric_key="mean_top1_score",
        round_digits=None,
    )

    assert summary["a"]["mean_true_label_score"] == pytest.approx(0.1111115)
    assert summary["a"]["mean_top1_score"] == pytest.approx(0.2222225)
    assert summary["a"]["mean_margin_top1_top2"] == pytest.approx(0.3333335)


def test_render_table_uses_requested_metric_headers() -> None:
    per_category = {
        "a": {
            "support": 2,
            "predicted": 2,
            "correct": 1,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "mean_true_label_score": 0.55,
            "mean_top_1_score": 0.8,
            "mean_margin_top1_top2": 0.25,
        }
    }

    confusion = render_confusion_table({"a": {"a": 1}})
    table = render_per_category_table(
        per_category,
        primary_metric_key="mean_true_label_score",
        top_1_metric_key="mean_top_1_score",
        primary_header="mean_true_score",
        top_1_header="mean_top1_score",
    )

    assert "actual \\ predicted" in confusion
    assert "mean_true_score" in table
    assert "mean_top1_score" in table


def test_build_classification_evaluation_report_exposes_paper_metrics() -> None:
    report = build_classification_evaluation_report(
        categories=["a", "b"],
        actual_labels=["a", "a", "b"],
        predicted_labels=["a", "b", "b"],
        true_probs=[0.9, 0.2, 0.8],
        top_1_values=[0.9, 0.7, 0.8],
        margins=[0.4, 0.1, 0.3],
        total_loss=1.25,
        total_rows=3,
    )

    assert report["rows_total"] == 3
    assert report["loss"] == pytest.approx(0.416667)
    assert report["accuracy_top_1"] == pytest.approx(0.666667)
    assert report["macro_f1"] == pytest.approx(0.666667)
    assert report["weighted_f1"] == pytest.approx(0.666667)
    assert report["worst_category_f1"] == "a"
    assert "max_calibration_error" in report

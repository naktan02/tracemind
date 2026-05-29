from __future__ import annotations

import pytest
import torch
from torch import nn

from scripts.support.query_ssl_peft.teacher_providers.fixed_embedding_classifier.evaluation import (
    evaluate_classifier,
    print_evaluation_report,
)


def test_evaluate_classifier_builds_report_and_prints_summary(capsys) -> None:
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.bias.zero_()

    report = evaluate_classifier(
        model=model,
        features=torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
        targets=torch.tensor([0, 1]),
        categories=["anxiety", "depression"],
        eval_batch_size=1,
        device="cpu",
    )

    assert report["rows_total"] == 2
    assert report["correct_top_1"] == 2
    assert report["accuracy_top_1"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["weighted_f1"] == 1.0
    assert report["balanced_accuracy"] == 1.0
    assert report["expected_calibration_error"] == pytest.approx(0.083314)
    assert report["mean_true_label_probability"] == pytest.approx(
        0.916686,
        rel=1e-6,
    )
    assert report["confusion_matrix"] == {
        "anxiety": {"anxiety": 1, "depression": 0},
        "depression": {"anxiety": 0, "depression": 1},
    }

    print_evaluation_report(dataset_name="validation", report=report)

    output = capsys.readouterr().out
    assert "[validation] accuracy_top_1=1.0000 macro_f1=1.0000" in output
    assert "ece=0.0833 rows=2" in output
    assert "| actual \\ predicted | anxiety | depression |" in output
    assert "| category | support | precision | recall | f1 |" in output

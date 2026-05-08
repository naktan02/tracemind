"""Fixed classifier 평가 계산과 출력 helper."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from methods.adaptation.common.classification_evaluation import (
    build_classification_evaluation_report,
)
from shared.src.domain.services.classification_report import (
    render_confusion_table,
    render_per_category_table,
)


def evaluate_classifier(
    *,
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    categories: list[str],
    eval_batch_size: int,
    device: str,
) -> dict[str, Any]:
    """분류 결과 지표를 계산한다."""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    actual_labels: list[str] = []
    predicted_labels: list[str] = []
    true_probs: list[float] = []
    top_1_probs: list[float] = []
    margins: list[float] = []
    total_loss = 0.0
    total_rows = len(targets)

    with torch.no_grad():
        for start in range(0, total_rows, eval_batch_size):
            end = min(start + eval_batch_size, total_rows)
            batch_features = features[start:end].to(device)
            batch_targets = targets[start:end].to(device)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            probabilities = torch.softmax(logits, dim=-1)
            top_values, top_indices = torch.topk(probabilities, k=2, dim=-1)
            predicted = top_indices[:, 0]
            true_probability = probabilities.gather(
                1,
                batch_targets.unsqueeze(1),
            ).squeeze(1)

            total_loss += float(loss.item()) * len(batch_targets)
            actual_labels.extend(
                categories[index] for index in batch_targets.cpu().tolist()
            )
            predicted_labels.extend(
                categories[index] for index in predicted.cpu().tolist()
            )
            true_probs.extend(true_probability.cpu().tolist())
            top_1_probs.extend(top_values[:, 0].cpu().tolist())
            margins.extend((top_values[:, 0] - top_values[:, 1]).cpu().tolist())

    return build_classification_evaluation_report(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        true_probs=true_probs,
        top_1_values=top_1_probs,
        margins=margins,
        total_loss=total_loss,
        total_rows=total_rows,
    )


def print_evaluation_report(
    *,
    dataset_name: str,
    report: dict[str, Any],
) -> None:
    """fixed classifier 평가 report를 콘솔에 출력한다."""

    print(
        f"[{dataset_name}] "
        f"accuracy_top_1={report['accuracy_top_1']:.4f} "
        f"rows={report['rows_total']} "
        f"mean_true_prob={report['mean_true_label_probability']:.4f} "
        f"mean_margin={report['mean_margin_top1_top2']:.4f}",
        flush=True,
    )
    print(render_confusion_table(report["confusion_matrix"]))
    print()
    print(
        render_per_category_table(
            report["per_category"],
            primary_metric_key="mean_true_label_probability",
            top_1_metric_key="mean_top_1_probability",
            primary_header="mean_true_prob",
            top_1_header="mean_top1_prob",
        )
    )
    print()

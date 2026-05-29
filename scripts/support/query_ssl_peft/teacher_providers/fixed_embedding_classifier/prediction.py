"""Fixed classifier prediction helper."""

from __future__ import annotations

import torch

from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

from .models import FixedClassifierPrediction, TrainedFixedClassifier
from .row_embeddings import embed_rows


def predict_fixed_classifier_rows(
    *,
    trained: TrainedFixedClassifier,
    rows: list[LabeledQueryRow],
    embed_chunk_size: int,
    eval_batch_size: int,
) -> list[FixedClassifierPrediction]:
    """학습된 fixed classifier로 unlabeled row를 추론한다."""

    features = embed_rows(
        rows=rows,
        adapter=trained.adapter,
        chunk_size=embed_chunk_size,
    )
    predictions: list[FixedClassifierPrediction] = []
    trained.model.eval()

    with torch.no_grad():
        for start in range(0, len(rows), eval_batch_size):
            end = min(start + eval_batch_size, len(rows))
            batch_rows = rows[start:end]
            batch_features = features[start:end].to(trained.training_device)
            logits = trained.model(batch_features)
            probabilities = torch.softmax(logits, dim=-1)
            top_values, top_indices = torch.topk(probabilities, k=2, dim=-1)
            for row, probability_row, top_value_row, top_index_row in zip(
                batch_rows,
                probabilities.cpu(),
                top_values.cpu(),
                top_indices.cpu(),
                strict=True,
            ):
                top1_index = int(top_index_row[0].item())
                top2_index = int(top_index_row[1].item())
                raw_scores = {
                    category: float(probability_row[index].item())
                    for index, category in enumerate(trained.categories)
                }
                predictions.append(
                    FixedClassifierPrediction(
                        query_id=str(row["query_id"]),
                        predicted_label=trained.categories[top1_index],
                        confidence=float(top_value_row[0].item()),
                        margin=float((top_value_row[0] - top_value_row[1]).item()),
                        runner_up_label=trained.categories[top2_index],
                        runner_up_score=float(top_value_row[1].item()),
                        raw_scores=raw_scores,
                    )
                )
    return predictions

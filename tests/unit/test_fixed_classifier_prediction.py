from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.support.query_ssl_peft.teacher_providers.fixed_embedding_classifier.models import (
    TrainedFixedClassifier,
)
from scripts.support.query_ssl_peft.teacher_providers.fixed_embedding_classifier.prediction import (
    predict_fixed_classifier_rows,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


class _FakeEmbeddingAdapter:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [2.0, 0.0] if text == "left class text" else [0.0, 3.0] for text in texts
        ]


def _row(query_id: str, label: str, text: str) -> LabeledQueryRow:
    return LabeledQueryRow(
        query_id=query_id,
        text=text,
        raw_label_scheme="manual_label",
        raw_label=label,
        mapped_label_4=label,
        locale="ko-KR",
        annotation_source="annotated",
        approved_by="annotator",
        created_at="2026-04-22T00:00:00+00:00",
    )


def test_predict_fixed_classifier_rows_returns_ranked_teacher_predictions() -> None:
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.bias.zero_()

    trained = TrainedFixedClassifier(
        model=model,
        adapter=_FakeEmbeddingAdapter(),
        embedding_spec=SimpleNamespace(),
        categories=["anxiety", "depression"],
        label_to_index={"anxiety": 0, "depression": 1},
        training_device="cpu",
        history=[],
        best_selection_report={},
        eval_results={},
    )

    predictions = predict_fixed_classifier_rows(
        trained=trained,
        rows=[
            _row("q1", "anxiety", "left class text"),
            _row("q2", "depression", "right class text"),
        ],
        embed_chunk_size=1,
        eval_batch_size=1,
    )

    assert [prediction.query_id for prediction in predictions] == ["q1", "q2"]
    assert [prediction.predicted_label for prediction in predictions] == [
        "anxiety",
        "depression",
    ]
    assert predictions[0].runner_up_label == "depression"
    assert predictions[1].runner_up_label == "anxiety"
    assert predictions[0].confidence == pytest.approx(0.880797, rel=1e-5)
    assert predictions[1].confidence == pytest.approx(0.952574, rel=1e-5)
    assert (
        predictions[0].raw_scores["anxiety"] > predictions[0].raw_scores["depression"]
    )

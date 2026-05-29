from __future__ import annotations

import torch

from scripts.support.query_ssl_peft.teacher_providers.fixed_embedding_classifier.training import (
    build_label_index,
    labels_to_tensor,
    train_classifier_head,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


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


def test_build_label_index_and_labels_to_tensor_are_deterministic() -> None:
    rows = [
        _row("q1", "depression", "우울해요"),
        _row("q2", "anxiety", "불안해요"),
        _row("q3", "anxiety", "걱정돼요"),
    ]

    categories, label_to_index = build_label_index(rows)
    targets = labels_to_tensor(rows, label_to_index)

    assert categories == ["anxiety", "depression"]
    assert label_to_index == {"anxiety": 0, "depression": 1}
    assert targets.tolist() == [1, 0, 0]


def test_train_classifier_head_tracks_selection_and_restores_best() -> None:
    torch.manual_seed(7)

    features = torch.tensor(
        [
            [4.0, 0.0],
            [3.0, 0.0],
            [0.0, 4.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    model, history, best_selection_report = train_classifier_head(
        train_features=features,
        train_targets=targets,
        selection_features=features,
        selection_targets=targets,
        categories=["anxiety", "depression"],
        training_device="cpu",
        epochs=8,
        train_batch_size=2,
        learning_rate=0.05,
        weight_decay=0.0,
    )

    assert len(history) == 8
    assert best_selection_report["rows_total"] == 4
    assert best_selection_report["accuracy_top_1"] == 1.0
    assert model.training is False

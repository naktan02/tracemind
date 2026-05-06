"""LoRA + classifier 공통 core 단위 검증."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from methods.adaptation.lora_classifier.modeling import LoraTextClassifier
from methods.adaptation.lora_classifier.training import (
    evaluate_classifier,
    train_classifier,
)
from methods.adaptation.query_classifier_adaptation import (
    modeling as legacy_modeling,
)
from methods.adaptation.query_classifier_adaptation import (
    training as legacy_training,
)


class _TinyBackbone(nn.Module):
    """transformers output shape만 흉내 내는 작은 backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)
        self.embedding = nn.Embedding(8, 3)
        self.projection = nn.Linear(3, 3)

    def forward(self, *, input_ids, attention_mask):
        del attention_mask
        hidden = self.projection(self.embedding(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)


def _build_loader() -> DataLoader[dict[str, torch.Tensor]]:
    rows = [
        {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor(0),
        },
        {
            "input_ids": torch.tensor([3, 4]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor(1),
        },
    ]
    return DataLoader(rows, batch_size=2)


def test_lora_text_classifier_train_step_and_evaluation() -> None:
    torch.manual_seed(7)
    model = LoraTextClassifier(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    loader = _build_loader()

    trained, history, selection_report = train_classifier(
        model=model,
        train_loader=loader,
        selection_loader=loader,
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=1,
        learning_rate=0.01,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
    )
    evaluation = evaluate_classifier(
        model=trained,
        dataloader=loader,
        categories=["anxiety", "normal"],
        device="cpu",
    )

    assert history[0]["epoch"] == 1
    assert selection_report["rows_total"] == 2
    assert evaluation["rows_total"] == 2
    assert set(evaluation["per_category"]) == {"anxiety", "normal"}


def test_query_classifier_adaptation_paths_remain_compatibility_shims() -> None:
    assert legacy_modeling.LoraTextClassifier is LoraTextClassifier
    assert legacy_training.train_classifier is train_classifier

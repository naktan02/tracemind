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
    train_query_ssl_classifier,
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


class _CountingQuerySslAlgorithm:
    algorithm_name = "counting"

    def __init__(self) -> None:
        self.steps = 0
        self.num_train_iter = 0

    @property
    def uses_labeled_batches(self) -> bool:
        return True

    def configure_training(self, *, num_train_iter: int) -> None:
        self.num_train_iter = num_train_iter

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        assert train_loader_length > 0
        assert unlabeled_loader_length > 0

    def compute_step(self, *, model, labeled_batch, unlabeled_batch):
        del unlabeled_batch
        self.steps += 1
        assert labeled_batch is not None
        logits = model(
            input_ids=labeled_batch["input_ids"],
            attention_mask=labeled_batch["attention_mask"],
        )
        loss = logits.mean()
        return SimpleNamespace(
            total_loss=loss,
            loss_components={"dummy_loss": loss.detach()},
            metrics={"dummy_metric": loss.detach()},
        )


def _build_unlabeled_loader() -> DataLoader[dict[str, torch.Tensor]]:
    rows = [
        {
            "weak_input_ids": torch.tensor([1, 2]),
            "weak_attention_mask": torch.tensor([1, 1]),
            "strong_input_ids": torch.tensor([2, 3]),
            "strong_attention_mask": torch.tensor([1, 1]),
        }
        for _ in range(5)
    ]
    return DataLoader(rows, batch_size=1)


def test_query_ssl_training_stops_at_max_train_steps() -> None:
    torch.manual_seed(7)
    model = LoraTextClassifier(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    algorithm = _CountingQuerySslAlgorithm()

    _, history, _ = train_query_ssl_classifier(
        model=model,
        train_loader=_build_loader(),
        unlabeled_loader=_build_unlabeled_loader(),
        selection_loader=_build_loader(),
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=5,
        max_train_steps=3,
        learning_rate=0.01,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
        algorithm=algorithm,
    )

    assert algorithm.num_train_iter == 3
    assert algorithm.steps == 3
    assert len(history) == 3


def test_query_ssl_training_resume_checkpoint_continues_remaining_steps(
    tmp_path,
) -> None:
    torch.manual_seed(7)
    checkpoint_dir = tmp_path / "checkpoints"
    model = LoraTextClassifier(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    first_algorithm = _CountingQuerySslAlgorithm()

    _, first_history, _ = train_query_ssl_classifier(
        model=model,
        train_loader=_build_loader(),
        unlabeled_loader=_build_unlabeled_loader(),
        selection_loader=_build_loader(),
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=2,
        max_train_steps=2,
        learning_rate=0.01,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
        algorithm=first_algorithm,
        resume_checkpoint_output_dir=checkpoint_dir,
        resume_checkpoint_every_epochs=1,
    )
    checkpoint_path = checkpoint_dir / "latest_training_checkpoint.pt"

    resumed_model = LoraTextClassifier(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    resumed_algorithm = _CountingQuerySslAlgorithm()
    _, resumed_history, _ = train_query_ssl_classifier(
        model=resumed_model,
        train_loader=_build_loader(),
        unlabeled_loader=_build_unlabeled_loader(),
        selection_loader=_build_loader(),
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=3,
        max_train_steps=3,
        learning_rate=0.01,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
        algorithm=resumed_algorithm,
        resume_checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()
    assert first_algorithm.steps == 2
    assert len(first_history) == 2
    assert resumed_algorithm.steps == 1
    assert len(resumed_history) == 3


def test_query_classifier_adaptation_paths_remain_compatibility_shims() -> None:
    assert legacy_modeling.LoraTextClassifier is LoraTextClassifier
    assert legacy_training.train_classifier is train_classifier

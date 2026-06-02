"""Reusable MeanTeacher method core tests."""

from __future__ import annotations

import torch
from torch import nn

from methods.ssl.algorithms.meanteacher.meanteacher import (
    MeanTeacherAlgorithm,
    compute_meanteacher_step,
)
from methods.ssl.registry import (
    build_query_ssl_algorithm,
    resolve_query_ssl_algorithm_descriptor,
)
from methods.ssl.runtime.ema import EmaTrainableParameterTeacher


class _TokenSumClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))

    def forward(self, *, input_ids, attention_mask):
        values = (input_ids.float() * attention_mask.float()).sum(dim=1)
        features = torch.stack([values, -values], dim=1)
        return features @ self.weight


class _TrainableFeatureClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 1024, bias=False)
        self.classifier = nn.Linear(1024, 4, bias=False)

    def forward(self, *, input_ids, attention_mask):
        del attention_mask
        features = self.encoder(input_ids.float())
        return self.classifier(features)


def _labeled_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }


def _unlabeled_batch() -> dict[str, torch.Tensor]:
    return {
        "weak_input_ids": torch.tensor([[1, 1], [0, 1]], dtype=torch.long),
        "weak_attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        "strong_input_ids": torch.tensor([[1, 0], [1, 1]], dtype=torch.long),
        "strong_attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
    }


def test_query_ssl_algorithm_registry_builds_meanteacher_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="mean_teacher",
        parameters={
            "ema_m": 0.9,
            "unsup_warm_up": 0.4,
            "lambda_u": 1.0,
        },
    )

    assert isinstance(algorithm, MeanTeacherAlgorithm)
    assert algorithm.algorithm_name == "meanteacher"
    assert algorithm.ema_m == 0.9


def test_query_ssl_algorithm_descriptor_exposes_meanteacher_view_spec() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("meanteacher")

    assert descriptor.algorithm_name == "meanteacher"
    assert descriptor.display_name == "MeanTeacher"
    assert descriptor.required_views.view_names == ("text", "aug_0", "aug_1")
    assert descriptor.required_views.view_builder_name == "usb_multiview"
    assert descriptor.default_uses_labeled_batches is True


def test_compute_meanteacher_step_uses_teacher_weak_student_strong_mse() -> None:
    model = _TokenSumClassifier()
    teacher = EmaTrainableParameterTeacher(model=model, momentum=0.5)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -0.25], [-0.5, 0.25]]))

    output = compute_meanteacher_step(
        model=model,
        ema_teacher=teacher,
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
        iteration=2,
        num_train_iter=10,
        unsup_warm_up=0.4,
        lambda_u=1.0,
    )

    with torch.no_grad():
        current_logits = model(
            input_ids=_labeled_batch()["input_ids"],
            attention_mask=_labeled_batch()["attention_mask"],
        )
        strong_logits = model(
            input_ids=_unlabeled_batch()["strong_input_ids"],
            attention_mask=_unlabeled_batch()["strong_attention_mask"],
        )
        original_weight = model.weight.detach().clone()
        model.weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
        weak_logits = model(
            input_ids=_unlabeled_batch()["weak_input_ids"],
            attention_mask=_unlabeled_batch()["weak_attention_mask"],
        )
        model.weight.copy_(original_weight)
    expected_sup_loss = nn.functional.cross_entropy(
        current_logits,
        _labeled_batch()["labels"],
        reduction="mean",
    )
    expected_unsup_loss = nn.functional.mse_loss(
        torch.softmax(strong_logits, dim=-1),
        torch.softmax(weak_logits.detach(), dim=-1),
        reduction="mean",
    )

    assert torch.isclose(output.loss_components["sup_loss"], expected_sup_loss)
    assert torch.isclose(output.loss_components["unsup_loss"], expected_unsup_loss)
    assert torch.isclose(output.metrics["unsup_warmup"], torch.tensor(0.5))
    assert torch.isclose(
        output.total_loss,
        expected_sup_loss + expected_unsup_loss * 0.5,
    )


def test_compute_meanteacher_step_backpropagates_through_trainable_features() -> None:
    model = _TrainableFeatureClassifier()
    teacher = EmaTrainableParameterTeacher(model=model, momentum=0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1)

    output = compute_meanteacher_step(
        model=model,
        ema_teacher=teacher,
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
        iteration=2,
        num_train_iter=10,
        unsup_warm_up=0.4,
        lambda_u=1.0,
    )
    output.total_loss.backward()

    assert model.encoder.weight.grad is not None
    assert model.classifier.weight.grad is not None


def test_meanteacher_algorithm_state_roundtrips_iteration_and_ema() -> None:
    model = _TokenSumClassifier()
    algorithm = MeanTeacherAlgorithm(ema_m=0.5, unsup_warm_up=0.4)
    algorithm.configure_training(num_train_iter=10)
    algorithm.configure_model(model=model, device=torch.device("cpu"))
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 1.0], [0.0, -2.0]]))
    algorithm.after_optimizer_step(
        model=model,
        step_context=object(),  # type: ignore[arg-type]
    )
    state = algorithm.export_state()

    restored = MeanTeacherAlgorithm(ema_m=0.5, unsup_warm_up=0.4)
    restored.configure_training(num_train_iter=10)
    restored.configure_model(model=model, device=torch.device("cpu"))
    restored.load_state(state)
    restored_state = restored.export_state()

    assert restored_state["iteration"] == state["iteration"]
    assert restored_state["ema_parameter_names"] == state["ema_parameter_names"]
    assert torch.allclose(
        restored_state["ema_shadow"]["weight"],
        state["ema_shadow"]["weight"],
    )

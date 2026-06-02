"""Reusable MixMatch method core tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from methods.ssl.algorithms.mixmatch.mixmatch import (
    MixMatchAlgorithm,
    compute_mixmatch_step,
    sharpen_probabilities,
)
from methods.ssl.algorithms.mixmatch.mixup import mixup_one_target
from methods.ssl.base import QuerySslStepContext
from methods.ssl.registry import (
    build_query_ssl_algorithm,
    resolve_query_ssl_algorithm_descriptor,
)


class _FeatureClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Identity()
        self.classifier = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.eye(2))

    def extract_pooled_features(self, *, input_ids, attention_mask):
        del attention_mask
        values = input_ids.float() * self.feature_scale
        return torch.stack([values[:, 0], values[:, 1]], dim=1)

    def forward(self, *, input_ids, attention_mask):
        pooled = self.extract_pooled_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.classifier(self.dropout(pooled))


def _labeled_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[2, 0], [0, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }


def _unlabeled_batch() -> dict[str, torch.Tensor]:
    return {
        "weak_input_ids": torch.tensor([[2, 0], [0, 2]], dtype=torch.long),
        "weak_attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        "strong_input_ids": torch.tensor([[1, 1], [2, 0]], dtype=torch.long),
        "strong_attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
    }


def _step_context(*, global_step: int = 1) -> QuerySslStepContext:
    return QuerySslStepContext(
        epoch_index=1,
        step_index=global_step,
        global_step=global_step,
        total_train_steps=64,
        num_classes=2,
        device=torch.device("cpu"),
    )


def test_query_ssl_algorithm_registry_builds_mixmatch_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="mixmatch",
        parameters={
            "T": 0.5,
            "unsup_warm_up": 0.25,
            "mixup_alpha": 0.7,
        },
    )

    assert isinstance(algorithm, MixMatchAlgorithm)
    assert algorithm.algorithm_name == "mixmatch"
    assert algorithm.T == 0.5
    assert algorithm.unsup_warm_up == 0.25
    assert algorithm.mixup_alpha == 0.7


def test_query_ssl_algorithm_descriptor_exposes_mixmatch_capability_spec() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("mixmatch")

    assert descriptor.algorithm_name == "mixmatch"
    assert descriptor.display_name == "MixMatch"
    assert descriptor.required_views.view_names == ("text", "aug_0", "aug_1")
    assert descriptor.required_views.view_builder_name == "usb_multiview"
    assert descriptor.runtime_requirements.step_context_required is True


def test_compute_mixmatch_step_uses_feature_mixup_and_warmup() -> None:
    model = _FeatureClassifier()

    output = compute_mixmatch_step(
        model=model,
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
        num_classes=2,
        temperature=0.5,
        unsup_warm_up=0.25,
        mixup_alpha=0.0,
        lambda_u=1.0,
        supervised_loss_weight=1.0,
        iteration=16,
        num_train_iter=64,
    )

    assert output.loss_components["sup_loss"] > 0
    assert output.loss_components["unsup_loss"] > 0
    assert output.metrics["unsup_warmup"] == 1.0
    assert output.metrics["effective_lambda_u"] == 1.0
    assert output.metrics["mixup_lambda"] == 1.0
    assert output.debug_tensors["mixed_targets"].shape == (6, 2)

    output.total_loss.backward()

    assert model.feature_scale.grad is not None
    assert model.feature_scale.grad.abs() > 0
    assert model.classifier.weight.grad is not None


def test_mixmatch_first_step_has_zero_unsupervised_weight_like_usb() -> None:
    algorithm = MixMatchAlgorithm(T=0.5, unsup_warm_up=0.25, mixup_alpha=0.0)
    algorithm.configure_dataset(num_classes=2, unlabeled_row_count=2)
    algorithm.configure_training(num_train_iter=64)

    output = algorithm.compute_step_with_context(
        model=_FeatureClassifier(),
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
        step_context=_step_context(global_step=1),
    )

    assert output.metrics["unsup_warmup"] == 0.0
    assert output.metrics["effective_lambda_u"] == 0.0


def test_mixmatch_requires_equal_labeled_and_unlabeled_batch_size() -> None:
    unlabeled_batch = _unlabeled_batch()
    unlabeled_batch["weak_input_ids"] = unlabeled_batch["weak_input_ids"][:1]
    unlabeled_batch["weak_attention_mask"] = unlabeled_batch["weak_attention_mask"][:1]
    unlabeled_batch["strong_input_ids"] = unlabeled_batch["strong_input_ids"][:1]
    unlabeled_batch["strong_attention_mask"] = unlabeled_batch["strong_attention_mask"][
        :1
    ]

    with pytest.raises(ValueError, match="unlabeled batch size"):
        compute_mixmatch_step(
            model=_FeatureClassifier(),
            labeled_batch=_labeled_batch(),
            unlabeled_batch=unlabeled_batch,
            num_classes=2,
            iteration=0,
            num_train_iter=64,
        )


def test_mixmatch_rejects_token_level_mixup_for_text_runtime() -> None:
    with pytest.raises(ValueError, match="mixup_manifold=True"):
        MixMatchAlgorithm(mixup_manifold=False)


def test_mixup_one_target_biases_lambda_like_usb() -> None:
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    mixed_inputs, mixed_targets, lam = mixup_one_target(
        inputs=inputs,
        targets=targets,
        alpha=0.5,
        bias_toward_primary=True,
    )

    assert mixed_inputs.shape == inputs.shape
    assert mixed_targets.shape == targets.shape
    assert 0.5 <= lam <= 1.0


def test_sharpen_probabilities_reduces_entropy() -> None:
    probs = torch.tensor([[0.7, 0.3]])

    sharpened = sharpen_probabilities(probs, temperature=0.5)

    assert torch.allclose(sharpened.sum(dim=-1), torch.ones(1))
    assert sharpened[0, 0] > probs[0, 0]

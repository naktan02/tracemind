"""Reusable ReMixMatch method core tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from methods.ssl.algorithms.remixmatch.remixmatch import (
    ReMixMatchAlgorithm,
    compute_remixmatch_step,
)
from methods.ssl.hooks.distribution_alignment import EmaDistributionAlignmentHook
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
        "strong_0_input_ids": torch.tensor([[1, 1], [2, 0]], dtype=torch.long),
        "strong_0_attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        "strong_1_input_ids": torch.tensor([[0, 2], [1, 1]], dtype=torch.long),
        "strong_1_attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
    }


def test_query_ssl_algorithm_registry_builds_remixmatch_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="remixmatch",
        parameters={
            "T": 0.5,
            "unsup_warm_up": 0.25,
            "mixup_alpha": 0.75,
            "kl_loss_ratio": 0.5,
            "rot_loss_ratio": 0.0,
        },
    )

    assert isinstance(algorithm, ReMixMatchAlgorithm)
    assert algorithm.algorithm_name == "remixmatch"
    assert algorithm.mixup_alpha == 0.75
    assert algorithm.kl_loss_ratio == 0.5


def test_query_ssl_algorithm_descriptor_exposes_remixmatch_capability_spec() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("remixmatch")

    assert descriptor.algorithm_name == "remixmatch"
    assert descriptor.display_name == "ReMixMatch"
    assert descriptor.required_views.view_names == ("text", "aug_0", "aug_1")
    assert descriptor.required_views.view_builder_name == "usb_weak_strong_pair"
    assert descriptor.runtime_requirements.step_context_required is True


def test_compute_remixmatch_step_uses_da_mixup_and_u1_loss() -> None:
    model = _FeatureClassifier()
    dist_align = EmaDistributionAlignmentHook(
        num_classes=2,
        momentum=0.0,
        p_target_type="gt",
        p_target=torch.tensor([0.75, 0.25], dtype=torch.float32),
    )

    output = compute_remixmatch_step(
        model=model,
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
        num_classes=2,
        dist_align_hook=dist_align,
        temperature=0.5,
        unsup_warm_up=0.25,
        mixup_alpha=0.0,
        kl_loss_ratio=0.5,
        lambda_u=1.0,
        supervised_loss_weight=1.0,
        iteration=16,
        num_train_iter=64,
    )

    assert output.loss_components["sup_loss"] > 0
    assert output.loss_components["unsup_loss"] > 0
    assert output.loss_components["u1_loss"] > 0
    assert output.metrics["unsup_warmup"] == 1.0
    assert output.metrics["effective_lambda_u"] == 1.0
    assert output.metrics["effective_lambda_kl"] == 0.5
    assert output.metrics["mixup_lambda"] == 1.0
    assert output.debug_tensors["mixed_targets"].shape == (8, 2)
    assert output.debug_tensors["aligned_prob_x_ulb"].shape == (2, 2)

    output.total_loss.backward()

    assert model.feature_scale.grad is not None
    assert model.feature_scale.grad.abs() > 0
    assert model.classifier.weight.grad is not None


def test_remixmatch_first_step_ramps_unsupervised_losses_like_usb() -> None:
    algorithm = ReMixMatchAlgorithm(
        T=0.5,
        unsup_warm_up=0.25,
        mixup_alpha=0.0,
        kl_loss_ratio=0.5,
        rot_loss_ratio=0.0,
    )
    algorithm.configure_dataset(num_classes=2, unlabeled_row_count=2)
    algorithm.configure_labeled_distribution(
        class_distribution=torch.tensor([0.75, 0.25], dtype=torch.float32)
    )
    algorithm.configure_training(num_train_iter=64)

    output = algorithm.compute_step_with_context(
        model=_FeatureClassifier(),
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
        step_context=type(
            "Context",
            (),
            {"global_step": 1, "total_train_steps": 64},
        )(),
    )

    assert output.metrics["unsup_warmup"] == 0.0
    assert output.metrics["effective_lambda_u"] == 0.0
    assert output.metrics["effective_lambda_kl"] == 0.0


def test_remixmatch_algorithm_state_roundtrips_distribution_alignment() -> None:
    algorithm = ReMixMatchAlgorithm(mixup_alpha=0.0)
    algorithm.configure_dataset(num_classes=2, unlabeled_row_count=2)
    algorithm.configure_labeled_distribution(
        class_distribution=torch.tensor([0.75, 0.25], dtype=torch.float32)
    )
    algorithm.compute_step(
        model=_FeatureClassifier(),
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
    )
    state = algorithm.export_state()

    restored = ReMixMatchAlgorithm(mixup_alpha=0.0)
    restored.configure_dataset(num_classes=2, unlabeled_row_count=2)
    restored.configure_labeled_distribution(
        class_distribution=torch.tensor([0.75, 0.25], dtype=torch.float32)
    )
    restored.load_state(state)
    restored_state = restored.export_state()

    assert torch.allclose(restored_state["p_target"], state["p_target"])
    assert torch.allclose(restored_state["p_model"], state["p_model"])


def test_remixmatch_rejects_image_rotation_loss_for_text_runtime() -> None:
    with pytest.raises(ValueError, match="rot_loss_ratio=0.0"):
        ReMixMatchAlgorithm(rot_loss_ratio=0.5)


def test_remixmatch_requires_equal_labeled_and_unlabeled_batch_size() -> None:
    unlabeled_batch = _unlabeled_batch()
    for key in (
        "weak_input_ids",
        "weak_attention_mask",
        "strong_0_input_ids",
        "strong_0_attention_mask",
        "strong_1_input_ids",
        "strong_1_attention_mask",
    ):
        unlabeled_batch[key] = unlabeled_batch[key][:1]

    with pytest.raises(ValueError, match="unlabeled batch size"):
        compute_remixmatch_step(
            model=_FeatureClassifier(),
            labeled_batch=_labeled_batch(),
            unlabeled_batch=unlabeled_batch,
            num_classes=2,
            dist_align_hook=EmaDistributionAlignmentHook(
                num_classes=2,
                momentum=0.0,
                p_target_type="gt",
                p_target=torch.tensor([0.75, 0.25], dtype=torch.float32),
            ),
            iteration=0,
            num_train_iter=64,
        )

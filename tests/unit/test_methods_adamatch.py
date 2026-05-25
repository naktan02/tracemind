"""Reusable AdaMatch method core tests."""

from __future__ import annotations

import torch
from torch import nn

from methods.ssl.algorithms.adamatch.adamatch import (
    AdaMatchAlgorithm,
    compute_adamatch_step,
)
from methods.ssl.hooks.adaptive_thresholding import RelativeConfidenceThresholdingHook
from methods.ssl.hooks.distribution_alignment import AdaMatchDistAlignHook
from methods.ssl.registry import (
    build_query_ssl_algorithm,
    resolve_query_ssl_algorithm_descriptor,
)


class _SequentialLogitModel(nn.Module):
    def __init__(self, outputs: list[torch.Tensor]) -> None:
        super().__init__()
        self._outputs = iter(outputs)

    def forward(self, *, input_ids, attention_mask):
        del input_ids, attention_mask
        return next(self._outputs)


def test_query_ssl_algorithm_registry_builds_adamatch_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="adamatch",
        parameters={
            "temperature": 0.5,
            "p_cutoff": 0.95,
            "hard_label": True,
            "ema_p": 0.999,
            "lambda_u": 1.0,
            "supervised_loss_weight": 1.0,
        },
    )

    assert isinstance(algorithm, AdaMatchAlgorithm)
    assert algorithm.algorithm_name == "adamatch"
    assert algorithm.p_cutoff == 0.95
    assert algorithm.ema_p == 0.999
    assert algorithm.uses_labeled_batches is True


def test_query_ssl_algorithm_descriptor_exposes_adamatch_view_spec() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("adamatch")

    assert descriptor.algorithm_name == "adamatch"
    assert descriptor.display_name == "AdaMatch"
    assert descriptor.required_views.view_names == ("text", "aug_0", "aug_1")
    assert descriptor.required_views.view_builder_name == "usb_multiview"
    assert descriptor.default_uses_labeled_batches is True


def test_adamatch_dist_align_hook_matches_usb_model_target_ema_update() -> None:
    hook = AdaMatchDistAlignHook(
        num_classes=2,
        momentum=0.5,
    )
    algorithm = AdaMatchAlgorithm(
        temperature=0.5,
        p_cutoff=0.95,
        ema_p=0.5,
    )
    probs_x_ulb = torch.tensor(
        [[0.8, 0.2], [0.4, 0.6]],
        dtype=torch.float32,
    )
    probs_x_lb = torch.tensor(
        [[0.9, 0.1], [0.7, 0.3]],
        dtype=torch.float32,
    )

    aligned = hook.dist_align(
        probs_x_ulb=probs_x_ulb,
        probs_x_lb=probs_x_lb,
        algorithm=algorithm,
    )

    expected_p_model = torch.tensor([0.6, 0.4], dtype=torch.float32)
    expected_p_target = torch.tensor([0.65, 0.35], dtype=torch.float32)
    expected_aligned = (
        probs_x_ulb * (expected_p_target + 1e-6) / (expected_p_model + 1e-6)
    )
    expected_aligned = expected_aligned / expected_aligned.sum(
        dim=-1,
        keepdim=True,
    )

    assert torch.allclose(hook.p_model, expected_p_model)
    assert torch.allclose(hook.p_target, expected_p_target)
    assert torch.allclose(aligned, expected_aligned)
    assert algorithm.p_model is hook.p_model
    assert algorithm.p_target is hook.p_target


def test_adamatch_thresholding_hook_matches_usb_relative_threshold() -> None:
    hook = RelativeConfidenceThresholdingHook()
    algorithm = AdaMatchAlgorithm(
        temperature=0.5,
        p_cutoff=0.8,
    )
    probs_x_lb = torch.tensor(
        [[0.9, 0.1], [0.7, 0.3]],
        dtype=torch.float32,
    )
    probs_x_ulb = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4]],
        dtype=torch.float32,
    )

    mask = hook.masking(
        algorithm,
        probs_x_lb=probs_x_lb,
        probs_x_ulb=probs_x_ulb,
    )

    assert torch.equal(mask, torch.tensor([1.0, 0.0]))


def test_compute_adamatch_step_matches_usb_dist_align_threshold_flow() -> None:
    model = _SequentialLogitModel(
        outputs=[
            torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32),
            torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32),
            torch.tensor([[2.0, 0.0], [0.2, 0.0]], dtype=torch.float32),
        ]
    )
    labeled_batch = {
        "input_ids": torch.ones((2, 2), dtype=torch.long),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }
    unlabeled_batch = {
        "weak_input_ids": torch.ones((2, 2), dtype=torch.long),
        "weak_attention_mask": torch.ones((2, 2), dtype=torch.long),
        "strong_input_ids": torch.ones((2, 2), dtype=torch.long),
        "strong_attention_mask": torch.ones((2, 2), dtype=torch.long),
    }
    algorithm = AdaMatchAlgorithm(
        temperature=0.5,
        p_cutoff=0.7,
        hard_label=True,
        ema_p=0.0,
        lambda_u=1.0,
        supervised_loss_weight=1.0,
    )
    algorithm.configure_dataset(num_classes=2, unlabeled_row_count=4)
    assert algorithm.dist_align_hook is not None

    output = compute_adamatch_step(
        model=model,  # type: ignore[arg-type]
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        temperature=0.5,
        p_cutoff=0.7,
        hard_label=True,
        lambda_u=1.0,
        supervised_loss_weight=1.0,
        dist_align_hook=algorithm.dist_align_hook,
        masking_hook=algorithm.masking_hook,
        algorithm=algorithm,
    )

    logits_x_lb = torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32)
    logits_x_ulb_s = torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32)
    logits_x_ulb_w = torch.tensor([[2.0, 0.0], [0.2, 0.0]], dtype=torch.float32)
    probs_x_lb = torch.softmax(logits_x_lb, dim=-1)
    probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
    expected_p_model = probs_x_ulb_w.mean(dim=0)
    expected_p_target = probs_x_lb.mean(dim=0)
    aligned_probs = (
        probs_x_ulb_w * (expected_p_target + 1e-6) / (expected_p_model + 1e-6)
    )
    aligned_probs = aligned_probs / aligned_probs.sum(dim=-1, keepdim=True)
    expected_mask = aligned_probs.max(dim=-1)[0].ge(
        probs_x_lb.max(dim=-1)[0].mean() * 0.7
    )
    expected_mask = expected_mask.to(torch.float32)
    expected_sup_loss = nn.functional.cross_entropy(
        logits_x_lb,
        torch.tensor([0, 1], dtype=torch.long),
        reduction="mean",
    )
    strong_losses = nn.functional.cross_entropy(
        logits_x_ulb_s,
        torch.argmax(aligned_probs, dim=-1),
        reduction="none",
    )
    expected_unsup_loss = (strong_losses * expected_mask).mean()

    assert torch.allclose(output.debug_tensors["mask"], expected_mask)
    assert torch.allclose(output.debug_tensors["p_model"], expected_p_model)
    assert torch.allclose(output.debug_tensors["p_target"], expected_p_target)
    assert torch.isclose(output.loss_components["sup_loss"], expected_sup_loss)
    assert torch.isclose(output.loss_components["unsup_loss"], expected_unsup_loss)
    assert torch.isclose(output.metrics["util_ratio"], expected_mask.mean())
    assert torch.isclose(output.total_loss, expected_sup_loss + expected_unsup_loss)


def test_adamatch_algorithm_requires_dataset_state_before_step() -> None:
    algorithm = AdaMatchAlgorithm(
        temperature=0.5,
        p_cutoff=0.95,
    )

    try:
        algorithm.compute_step(
            model=_SequentialLogitModel([]),  # type: ignore[arg-type]
            labeled_batch=None,
            unlabeled_batch={
                "weak_input_ids": torch.ones((1, 2), dtype=torch.long),
                "weak_attention_mask": torch.ones((1, 2), dtype=torch.long),
                "strong_input_ids": torch.ones((1, 2), dtype=torch.long),
                "strong_attention_mask": torch.ones((1, 2), dtype=torch.long),
            },
        )
    except ValueError as exc:
        assert "configure_dataset" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("AdaMatch should require dataset state.")


def test_adamatch_step_requires_labeled_batch_after_dataset_configuration() -> None:
    algorithm = AdaMatchAlgorithm(
        temperature=0.5,
        p_cutoff=0.95,
        supervised_loss_weight=0.0,
    )
    algorithm.configure_dataset(num_classes=2, unlabeled_row_count=4)
    assert algorithm.dist_align_hook is not None

    try:
        compute_adamatch_step(
            model=_SequentialLogitModel([]),  # type: ignore[arg-type]
            labeled_batch=None,
            unlabeled_batch={
                "weak_input_ids": torch.ones((1, 2), dtype=torch.long),
                "weak_attention_mask": torch.ones((1, 2), dtype=torch.long),
                "strong_input_ids": torch.ones((1, 2), dtype=torch.long),
                "strong_attention_mask": torch.ones((1, 2), dtype=torch.long),
            },
            temperature=0.5,
            p_cutoff=0.95,
            supervised_loss_weight=0.0,
            dist_align_hook=algorithm.dist_align_hook,
        )
    except ValueError as exc:
        assert "labeled_batch" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("AdaMatch should require labeled_batch.")

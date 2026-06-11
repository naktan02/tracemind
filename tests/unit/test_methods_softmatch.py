"""Reusable SoftMatch method core tests."""

from __future__ import annotations

import torch
from torch import nn

from methods.ssl.algorithms.softmatch.softmatch import (
    SoftMatchAlgorithm,
    compute_softmatch_step,
)
from methods.ssl.algorithms.softmatch.weighting import SoftMatchWeightingHook
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


def test_query_ssl_algorithm_registry_builds_softmatch_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="softmatch",
        parameters={
            "temperature": 0.5,
            "hard_label": True,
            "dist_align": True,
            "dist_uniform": True,
            "ema_p": 0.999,
            "n_sigma": 2.0,
            "per_class": False,
            "lambda_u": 1.0,
            "supervised_loss_weight": 1.0,
        },
    )

    assert isinstance(algorithm, SoftMatchAlgorithm)
    assert algorithm.algorithm_name == "softmatch"
    assert algorithm.dist_align is True
    assert algorithm.dist_uniform is True


def test_query_ssl_algorithm_descriptor_exposes_softmatch_capabilities() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("softmatch")
    requirements = descriptor.runtime_requirements

    assert descriptor.algorithm_name == "softmatch"
    assert descriptor.display_name == "SoftMatch"
    assert descriptor.required_views.view_names == ("text", "aug_0", "aug_1")
    assert descriptor.required_views.view_builder_name == "usb_multiview"
    assert requirements.batch_surface == "weak_strong"
    assert requirements.model_outputs == frozenset({"logits"})
    assert requirements.algorithm_state_surface == frozenset(
        {"distribution_ema", "weighting_ema"}
    )


def test_softmatch_weighting_hook_matches_usb_truncated_gaussian_update() -> None:
    hook = SoftMatchWeightingHook(
        num_classes=2,
        n_sigma=2.0,
        momentum=0.5,
        per_class=False,
    )
    probs = torch.tensor([[0.8, 0.2], [0.4, 0.6]], dtype=torch.float32)

    mask = hook.masking(logits_x_ulb=probs, softmax_x_ulb=False)

    max_probs = probs.max(dim=-1).values
    expected_mu = torch.tensor(0.5) * 0.5 + max_probs.mean() * 0.5
    expected_var = torch.tensor(1.0) * 0.5 + torch.var(max_probs, unbiased=True) * 0.5
    expected_mask = torch.exp(
        -(torch.clamp(max_probs - expected_mu, max=0.0) ** 2)
        / ((2 * expected_var) / (2.0**2))
    )

    assert hook.hook_name == "softmatch_weighting"
    assert torch.allclose(hook.prob_max_mu_t, expected_mu)
    assert torch.allclose(hook.prob_max_var_t, expected_var)
    assert torch.allclose(mask, expected_mask)


def test_compute_softmatch_step_uses_weighted_consistency_loss() -> None:
    model = _SequentialLogitModel(
        outputs=[
            torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32),
            torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32),
            torch.tensor([[6.0, 0.0], [0.8, 0.0]], dtype=torch.float32),
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
    weighting_hook = SoftMatchWeightingHook(
        num_classes=2,
        n_sigma=2.0,
        momentum=0.5,
    )

    output = compute_softmatch_step(
        model=model,  # type: ignore[arg-type]
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        temperature=0.5,
        hard_label=True,
        lambda_u=1.0,
        supervised_loss_weight=1.0,
        dist_align_hook=None,
        weighting_hook=weighting_hook,
    )

    weak_logits = torch.tensor([[6.0, 0.0], [0.8, 0.0]], dtype=torch.float32)
    weak_probs = torch.softmax(weak_logits, dim=-1)
    expected_sup_loss = nn.functional.cross_entropy(
        torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.long),
        reduction="mean",
    )
    strong_losses = nn.functional.cross_entropy(
        torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32),
        torch.argmax(weak_probs, dim=-1),
        reduction="none",
    )
    expected_unsup_loss = (strong_losses * output.debug_tensors["mask"]).mean()

    assert torch.isclose(output.loss_components["sup_loss"], expected_sup_loss)
    assert torch.isclose(output.loss_components["unsup_loss"], expected_unsup_loss)
    assert torch.isclose(output.total_loss, expected_sup_loss + expected_unsup_loss)
    assert torch.isclose(
        output.metrics["util_ratio"],
        output.debug_tensors["mask"].float().mean(),
    )


def test_softmatch_algorithm_state_roundtrips_ema_state() -> None:
    algorithm = SoftMatchAlgorithm(
        temperature=0.5,
        ema_p=0.5,
        n_sigma=2.0,
    )
    algorithm.configure_dataset(num_classes=2, unlabeled_row_count=8)
    assert algorithm.dist_align_hook is not None
    assert algorithm.weighting_hook is not None
    algorithm.dist_align_hook.dist_align(
        probs_x_ulb=torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float32),
    )
    algorithm.weighting_hook.masking(
        logits_x_ulb=torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32),
        softmax_x_ulb=False,
    )
    state = algorithm.export_state()

    restored = SoftMatchAlgorithm(
        temperature=0.5,
        ema_p=0.5,
        n_sigma=2.0,
    )
    restored.configure_dataset(num_classes=2, unlabeled_row_count=8)
    restored.load_state(state)

    assert restored.dist_align_hook is not None
    assert restored.weighting_hook is not None
    assert torch.allclose(
        restored.dist_align_hook.p_model, algorithm.dist_align_hook.p_model
    )
    assert torch.allclose(
        restored.dist_align_hook.p_target,
        algorithm.dist_align_hook.p_target,
    )
    assert torch.allclose(
        restored.weighting_hook.prob_max_mu_t,
        algorithm.weighting_hook.prob_max_mu_t,
    )
    assert torch.allclose(
        restored.weighting_hook.prob_max_var_t,
        algorithm.weighting_hook.prob_max_var_t,
    )

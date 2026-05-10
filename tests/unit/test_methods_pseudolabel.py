"""Reusable USB PseudoLabel method core tests."""

from __future__ import annotations

import torch
from torch import nn

from methods.ssl.algorithms.pseudolabel.pseudolabel import (
    PseudoLabelAlgorithm,
    compute_pseudolabel_step,
)
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


def test_query_ssl_algorithm_registry_builds_pseudolabel_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="pseudolabel",
        parameters={
            "p_cutoff": 0.95,
            "unsup_warm_up": 0.4,
            "lambda_u": 1.0,
            "supervised_loss_weight": 1.0,
        },
    )

    assert isinstance(algorithm, PseudoLabelAlgorithm)
    assert algorithm.algorithm_name == "pseudolabel"
    assert algorithm.p_cutoff == 0.95


def test_query_ssl_algorithm_descriptor_exposes_pseudolabel_view_spec() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("pseudo_label")

    assert descriptor.algorithm_name == "pseudolabel"
    assert descriptor.display_name == "PseudoLabel"
    assert descriptor.required_views.view_names == ("weak_text",)
    assert descriptor.required_views.view_builder_name == "usb_weak"
    assert descriptor.default_uses_labeled_batches is True


def test_compute_pseudolabel_step_matches_usb_masked_warmup_loss() -> None:
    logits_x_lb = torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32)
    logits_x_ulb = torch.tensor([[3.0, 0.0], [0.2, 0.0]], dtype=torch.float32)
    model = _SequentialLogitModel(outputs=[logits_x_lb, logits_x_ulb])
    labeled_batch = {
        "input_ids": torch.ones((2, 2), dtype=torch.long),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }
    unlabeled_batch = {
        "weak_input_ids": torch.ones((2, 2), dtype=torch.long),
        "weak_attention_mask": torch.ones((2, 2), dtype=torch.long),
    }

    output = compute_pseudolabel_step(
        model=model,  # type: ignore[arg-type]
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        p_cutoff=0.7,
        unsup_warm_up=0.4,
        lambda_u=1.0,
        supervised_loss_weight=1.0,
        iteration=2,
        num_train_iter=10,
    )

    expected_sup_loss = nn.functional.cross_entropy(
        logits_x_lb,
        torch.tensor([0, 1], dtype=torch.long),
        reduction="mean",
    )
    expected_mask = torch.tensor([1.0, 0.0], dtype=torch.float32)
    unsup_losses = nn.functional.cross_entropy(
        logits_x_ulb,
        torch.argmax(logits_x_ulb.detach(), dim=-1),
        reduction="none",
    )
    expected_unsup_loss = (unsup_losses * expected_mask).mean()
    expected_warmup = torch.tensor(0.5)

    assert torch.equal(output.mask, expected_mask)
    assert torch.isclose(output.sup_loss, expected_sup_loss)
    assert torch.isclose(output.unsup_loss, expected_unsup_loss)
    assert torch.isclose(output.unsup_warmup, expected_warmup)
    assert torch.isclose(
        output.total_loss,
        expected_sup_loss + expected_unsup_loss * expected_warmup,
    )
    assert output.loss_components["sup_loss"] is output.sup_loss
    assert output.loss_components["unsup_loss"] is output.unsup_loss
    assert torch.isclose(output.metrics["util_ratio"], output.util_ratio)
    assert torch.isclose(output.metrics["unsup_warmup"], output.unsup_warmup)


def test_pseudolabel_algorithm_uses_usb_iteration_before_increment() -> None:
    first_logits_x_lb = torch.tensor([[4.0, 0.0]], dtype=torch.float32)
    first_logits_x_ulb = torch.tensor([[4.0, 0.0]], dtype=torch.float32)
    second_logits_x_lb = torch.tensor([[4.0, 0.0]], dtype=torch.float32)
    second_logits_x_ulb = torch.tensor([[4.0, 0.0]], dtype=torch.float32)
    model = _SequentialLogitModel(
        outputs=[
            first_logits_x_lb,
            first_logits_x_ulb,
            second_logits_x_lb,
            second_logits_x_ulb,
        ]
    )
    batch = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "labels": torch.tensor([0], dtype=torch.long),
    }
    unlabeled_batch = {
        "weak_input_ids": torch.ones((1, 2), dtype=torch.long),
        "weak_attention_mask": torch.ones((1, 2), dtype=torch.long),
    }
    algorithm = PseudoLabelAlgorithm(p_cutoff=0.95, unsup_warm_up=0.5)
    algorithm.configure_training(num_train_iter=4)

    first_output = algorithm.compute_step(
        model=model,  # type: ignore[arg-type]
        labeled_batch=batch,
        unlabeled_batch=unlabeled_batch,
    )
    second_output = algorithm.compute_step(
        model=model,  # type: ignore[arg-type]
        labeled_batch=batch,
        unlabeled_batch=unlabeled_batch,
    )

    assert torch.isclose(first_output.unsup_warmup, torch.tensor(0.0))
    assert torch.isclose(second_output.unsup_warmup, torch.tensor(0.5))

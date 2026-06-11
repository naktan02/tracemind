"""Reusable ReFixMatch method core tests."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from methods.ssl.algorithms.refixmatch.refixmatch import (
    ReFixMatchAlgorithm,
    compute_refixmatch_step,
    refixmatch_low_confidence_kl_loss,
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


def _labeled_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.ones((2, 2), dtype=torch.long),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }


def _unlabeled_batch() -> dict[str, torch.Tensor]:
    return {
        "weak_input_ids": torch.ones((2, 2), dtype=torch.long),
        "weak_attention_mask": torch.ones((2, 2), dtype=torch.long),
        "strong_input_ids": torch.ones((2, 2), dtype=torch.long),
        "strong_attention_mask": torch.ones((2, 2), dtype=torch.long),
    }


def test_query_ssl_algorithm_registry_builds_refixmatch_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="refixmatch",
        parameters={
            "temperature": 0.5,
            "p_cutoff": 0.95,
            "hard_label": True,
            "lambda_u": 1.0,
            "supervised_loss_weight": 1.0,
        },
    )

    assert isinstance(algorithm, ReFixMatchAlgorithm)
    assert algorithm.algorithm_name == "refixmatch"
    assert algorithm.p_cutoff == 0.95


def test_query_ssl_algorithm_descriptor_exposes_refixmatch_view_spec() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("refixmatch")

    assert descriptor.algorithm_name == "refixmatch"
    assert descriptor.display_name == "ReFixMatch"
    assert descriptor.required_views.view_names == ("text", "aug_0", "aug_1")
    assert descriptor.required_views.view_builder_name == "usb_multiview"
    assert descriptor.default_uses_labeled_batches is True


def test_compute_refixmatch_step_adds_low_confidence_kl_loss() -> None:
    strong_logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32)
    weak_logits = torch.tensor([[6.0, 0.0], [0.8, 0.0]], dtype=torch.float32)
    model = _SequentialLogitModel(
        outputs=[
            torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32),
            strong_logits,
            weak_logits,
        ]
    )

    output = compute_refixmatch_step(
        model=model,  # type: ignore[arg-type]
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
        temperature=0.5,
        p_cutoff=0.95,
        hard_label=True,
        lambda_u=1.0,
        supervised_loss_weight=1.0,
    )

    weak_probs = torch.softmax(weak_logits, dim=-1)
    expected_mask = torch.tensor([1.0, 0.0], dtype=torch.float32)
    expected_sup_loss = F.cross_entropy(
        torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.long),
        reduction="mean",
    )
    strong_losses = F.cross_entropy(
        strong_logits,
        torch.argmax(weak_probs, dim=-1),
        reduction="none",
    )
    expected_unsup_loss = (strong_losses * expected_mask).mean()
    expected_refix_loss = refixmatch_low_confidence_kl_loss(
        logits=strong_logits,
        targets=weak_probs,
        mask=expected_mask,
    )

    assert torch.allclose(output.debug_tensors["mask"], expected_mask)
    assert torch.isclose(output.loss_components["sup_loss"], expected_sup_loss)
    assert torch.isclose(output.loss_components["unsup_loss"], expected_unsup_loss)
    assert torch.isclose(output.loss_components["refix_loss"], expected_refix_loss)
    assert torch.isclose(
        output.total_loss,
        expected_sup_loss + expected_unsup_loss + expected_refix_loss,
    )


def test_refixmatch_kl_loss_uses_mask_complement_like_usb() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
    targets = torch.tensor([[0.95, 0.05], [0.55, 0.45]], dtype=torch.float32)
    mask = torch.tensor([1.0, 0.0], dtype=torch.float32)

    loss = refixmatch_low_confidence_kl_loss(
        logits=logits,
        targets=targets,
        mask=mask,
    )
    per_class = F.kl_div(
        F.log_softmax(logits / 0.5, dim=-1),
        F.softmax(targets / 0.5, dim=-1),
        reduction="none",
    )
    expected = torch.sum(
        per_class * (1.0 - mask).unsqueeze(-1).repeat(1, 2),
        dim=1,
    ).mean()

    assert torch.isclose(loss, expected)


def test_compute_refixmatch_step_supports_unlabeled_only_ablation() -> None:
    weak_logits = torch.tensor([[6.0, 0.0], [0.8, 0.0]], dtype=torch.float32)
    model = _SequentialLogitModel(
        outputs=[
            torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32),
            weak_logits,
        ]
    )

    output = compute_refixmatch_step(
        model=model,  # type: ignore[arg-type]
        labeled_batch=None,
        unlabeled_batch=_unlabeled_batch(),
        temperature=0.5,
        p_cutoff=0.95,
        hard_label=True,
        lambda_u=1.0,
        supervised_loss_weight=0.0,
    )

    assert torch.isclose(output.loss_components["sup_loss"], torch.tensor(0.0))
    assert output.loss_components["unsup_loss"] > 0
    assert output.loss_components["refix_loss"] > 0

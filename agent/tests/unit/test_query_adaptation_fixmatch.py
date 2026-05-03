"""Query adaptation FixMatch core tests."""

from __future__ import annotations

import torch
from torch import nn

from agent.src.services.training.query_adaptation.algorithms.common import (
    build_pseudo_label_from_probs,
)
from agent.src.services.training.query_adaptation.algorithms.fixmatch.algorithm import (
    FixMatchAlgorithm,
    FixMatchConfig,
    compute_fixmatch_step,
)
from agent.src.services.training.query_adaptation.algorithms.registry import (
    build_query_ssl_algorithm,
)


class _SequentialLogitModel(nn.Module):
    def __init__(self, outputs: list[torch.Tensor]) -> None:
        super().__init__()
        self._outputs = iter(outputs)

    def forward(self, *, input_ids, attention_mask):
        del input_ids, attention_mask
        return next(self._outputs)


def test_build_fixmatch_pseudo_label_keeps_usb_temperature_behavior() -> None:
    probs = torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float32)

    hard_targets = build_pseudo_label_from_probs(
        probs_x_ulb_w=probs,
        use_hard_label=True,
        temperature=0.5,
    )
    soft_targets = build_pseudo_label_from_probs(
        probs_x_ulb_w=probs,
        use_hard_label=False,
        temperature=0.1,
    )

    assert hard_targets.tolist() == [0, 0]
    assert torch.equal(soft_targets, probs)


def test_query_ssl_algorithm_registry_builds_fixmatch_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="fixmatch",
        parameters={
            "temperature": 0.5,
            "p_cutoff": 0.95,
            "hard_label": True,
            "lambda_u": 1.0,
            "supervised_loss_weight": 1.0,
        },
    )

    assert isinstance(algorithm, FixMatchAlgorithm)
    assert algorithm.algorithm_name == "fixmatch"
    assert algorithm.config.p_cutoff == 0.95


def test_compute_fixmatch_step_matches_usb_masked_consistency_loss() -> None:
    model = _SequentialLogitModel(
        outputs=[
            torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32),
            torch.tensor([[6.0, 0.0], [0.8, 0.0]], dtype=torch.float32),
            torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32),
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

    output = compute_fixmatch_step(
        model=model,  # type: ignore[arg-type]
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        config=FixMatchConfig(
            temperature=0.5,
            p_cutoff=0.95,
            hard_label=True,
            lambda_u=1.0,
            supervised_loss_weight=1.0,
        ),
    )

    weak_probs = torch.softmax(torch.tensor([[6.0, 0.0], [0.8, 0.0]]), dim=-1)
    expected_mask = torch.tensor([1.0, 0.0], dtype=torch.float32)
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
    expected_unsup_loss = (strong_losses * expected_mask).mean()

    assert torch.allclose(output.mask, expected_mask)
    assert torch.isclose(output.sup_loss, expected_sup_loss)
    assert torch.isclose(output.unsup_loss, expected_unsup_loss)
    assert output.loss_components["sup_loss"] is output.sup_loss
    assert output.loss_components["unsup_loss"] is output.unsup_loss
    assert torch.isclose(output.metrics["util_ratio"], output.util_ratio)
    assert torch.isclose(
        output.total_loss,
        expected_sup_loss + expected_unsup_loss,
    )


def test_compute_fixmatch_step_supports_unlabeled_only_ablation() -> None:
    model = _SequentialLogitModel(
        outputs=[
            torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32),
            torch.tensor([[6.0, 0.0], [0.8, 0.0]], dtype=torch.float32),
        ]
    )
    unlabeled_batch = {
        "weak_input_ids": torch.ones((2, 2), dtype=torch.long),
        "weak_attention_mask": torch.ones((2, 2), dtype=torch.long),
        "strong_input_ids": torch.ones((2, 2), dtype=torch.long),
        "strong_attention_mask": torch.ones((2, 2), dtype=torch.long),
    }

    output = compute_fixmatch_step(
        model=model,  # type: ignore[arg-type]
        labeled_batch=None,
        unlabeled_batch=unlabeled_batch,
        config=FixMatchConfig(
            temperature=0.5,
            p_cutoff=0.95,
            hard_label=True,
            lambda_u=1.0,
            supervised_loss_weight=0.0,
        ),
    )

    weak_probs = torch.softmax(torch.tensor([[6.0, 0.0], [0.8, 0.0]]), dim=-1)
    expected_mask = torch.tensor([1.0, 0.0], dtype=torch.float32)
    strong_losses = nn.functional.cross_entropy(
        torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32),
        torch.argmax(weak_probs, dim=-1),
        reduction="none",
    )
    expected_unsup_loss = (strong_losses * expected_mask).mean()

    assert torch.allclose(output.mask, expected_mask)
    assert torch.isclose(output.sup_loss, torch.tensor(0.0))
    assert torch.isclose(output.unsup_loss, expected_unsup_loss)
    assert torch.isclose(output.total_loss, expected_unsup_loss)

"""Reusable PiModel method core tests."""

from __future__ import annotations

import torch
from torch import nn

from methods.ssl.algorithms.pimodel.pimodel import (
    PiModelAlgorithm,
    compute_pimodel_step,
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


def test_query_ssl_algorithm_registry_builds_pimodel_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="pimodel",
        parameters={
            "unsup_warm_up": 0.4,
            "lambda_u": 1.0,
        },
    )

    assert isinstance(algorithm, PiModelAlgorithm)
    assert algorithm.algorithm_name == "pimodel"
    assert algorithm.unsup_warm_up == 0.4


def test_query_ssl_algorithm_descriptor_exposes_pimodel_view_spec() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("pi_model")

    assert descriptor.algorithm_name == "pimodel"
    assert descriptor.display_name == "PiModel"
    assert descriptor.required_views.view_names == ("text", "aug_0", "aug_1")
    assert descriptor.required_views.view_builder_name == "usb_multiview"
    assert descriptor.default_uses_labeled_batches is True


def test_compute_pimodel_step_matches_usb_mse_consistency_and_warmup() -> None:
    logits_x_lb = torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32)
    logits_x_ulb_w = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    logits_x_ulb_s = torch.tensor([[0.8, 0.0], [0.2, 0.0]], dtype=torch.float32)
    model = _SequentialLogitModel(
        outputs=[logits_x_lb, logits_x_ulb_w, logits_x_ulb_s]
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

    output = compute_pimodel_step(
        model=model,  # type: ignore[arg-type]
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        iteration=2,
        num_train_iter=10,
        unsup_warm_up=0.4,
        lambda_u=1.0,
    )

    expected_sup_loss = nn.functional.cross_entropy(
        logits_x_lb,
        labeled_batch["labels"],
        reduction="mean",
    )
    expected_unsup_loss = nn.functional.mse_loss(
        torch.softmax(logits_x_ulb_s, dim=-1),
        torch.softmax(logits_x_ulb_w.detach(), dim=-1),
        reduction="mean",
    )
    expected_warmup = torch.tensor(0.5)

    assert torch.isclose(output.loss_components["sup_loss"], expected_sup_loss)
    assert torch.isclose(output.loss_components["unsup_loss"], expected_unsup_loss)
    assert torch.isclose(output.metrics["unsup_warmup"], expected_warmup)
    assert torch.isclose(
        output.total_loss,
        expected_sup_loss + expected_unsup_loss * expected_warmup,
    )


def test_pimodel_algorithm_iteration_state_roundtrips() -> None:
    algorithm = PiModelAlgorithm(unsup_warm_up=0.4)
    algorithm.configure_training(num_train_iter=10)
    algorithm.load_state(
        {
            "schema_version": "query_ssl_algorithm_state.v1",
            "algorithm_name": "pimodel",
            "configured": True,
            "iteration": 4,
            "num_train_iter": 10,
        }
    )

    state = algorithm.export_state()

    assert state["iteration"] == 4
    assert state["num_train_iter"] == 10

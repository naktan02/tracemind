"""Reusable UDA method core tests."""

from __future__ import annotations

import torch
from torch import nn

from methods.ssl.algorithms.uda.uda import (
    UDAAlgorithm,
    compute_tsa_threshold,
    compute_uda_step,
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


def test_query_ssl_algorithm_registry_builds_uda_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="uda",
        parameters={
            "T": 0.4,
            "p_cutoff": 0.8,
            "tsa_schedule": "linear",
            "lambda_u": 1.0,
        },
    )

    assert isinstance(algorithm, UDAAlgorithm)
    assert algorithm.algorithm_name == "uda"
    assert algorithm.temperature == 0.4
    assert algorithm.p_cutoff == 0.8
    assert algorithm.tsa_schedule == "linear"


def test_query_ssl_algorithm_descriptor_exposes_uda_view_spec() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("uda")

    assert descriptor.algorithm_name == "uda"
    assert descriptor.display_name == "UDA"
    assert descriptor.required_views.view_names == ("text", "aug_0", "aug_1")
    assert descriptor.required_views.view_builder_name == "usb_multiview"
    assert descriptor.default_uses_labeled_batches is True


def test_compute_tsa_threshold_matches_usb_schedule() -> None:
    assert compute_tsa_threshold(
        schedule="none",
        iteration=3,
        num_train_iter=10,
        num_classes=2,
    ) == 1.0
    assert compute_tsa_threshold(
        schedule="linear",
        iteration=5,
        num_train_iter=10,
        num_classes=2,
    ) == 0.75
    assert round(
        compute_tsa_threshold(
            schedule="log",
            iteration=5,
            num_train_iter=10,
            num_classes=2,
        ),
        6,
    ) == 0.958958


def test_compute_uda_step_matches_usb_tsa_and_soft_consistency_flow() -> None:
    logits_x_lb = torch.tensor([[2.0, 0.0], [0.2, 0.0]], dtype=torch.float32)
    logits_x_ulb_s = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    logits_x_ulb_w = torch.tensor([[2.0, 0.0], [0.1, 0.0]], dtype=torch.float32)
    model = _SequentialLogitModel(
        outputs=[logits_x_lb, logits_x_ulb_s, logits_x_ulb_w]
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

    output = compute_uda_step(
        model=model,  # type: ignore[arg-type]
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        temperature=0.4,
        p_cutoff=0.8,
        tsa_schedule="linear",
        iteration=5,
        num_train_iter=10,
        num_classes=2,
        lambda_u=1.0,
    )

    expected_tsa = torch.tensor(0.75)
    expected_sup_mask = torch.tensor([0.0, 1.0])
    expected_sup_loss = (
        nn.functional.cross_entropy(
            logits_x_lb,
            labeled_batch["labels"],
            reduction="none",
        )
        * expected_sup_mask
    ).mean()
    weak_probs = torch.softmax(logits_x_ulb_w, dim=-1)
    expected_mask = torch.tensor([1.0, 0.0])
    log_probs = torch.log_softmax(logits_x_ulb_s, dim=-1)
    expected_unsup_loss = (-(weak_probs * log_probs).sum(dim=-1) * expected_mask).mean()

    assert torch.allclose(output.metrics["tsa"], expected_tsa)
    assert torch.equal(output.debug_tensors["sup_mask"], expected_sup_mask)
    assert torch.equal(output.debug_tensors["mask"], expected_mask)
    assert torch.equal(output.debug_tensors["pseudo_label"], weak_probs)
    assert torch.isclose(output.loss_components["sup_loss"], expected_sup_loss)
    assert torch.isclose(output.loss_components["unsup_loss"], expected_unsup_loss)
    assert torch.isclose(output.metrics["util_ratio"], expected_mask.mean())
    assert torch.isclose(output.total_loss, expected_sup_loss + expected_unsup_loss)


def test_uda_algorithm_iteration_state_roundtrips() -> None:
    algorithm = UDAAlgorithm(temperature=0.4, p_cutoff=0.8, tsa_schedule="linear")
    algorithm.configure_dataset(num_classes=2, unlabeled_row_count=4)
    algorithm.configure_training(num_train_iter=10)
    algorithm.load_state(
        {
            "schema_version": "query_ssl_algorithm_state.v1",
            "algorithm_name": "uda",
            "configured": True,
            "iteration": 4,
            "num_train_iter": 10,
            "num_classes": 2,
        }
    )

    state = algorithm.export_state()

    assert state["iteration"] == 4
    assert state["num_train_iter"] == 10
    assert state["num_classes"] == 2

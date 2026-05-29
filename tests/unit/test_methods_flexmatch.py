"""Reusable FlexMatch method core tests."""

from __future__ import annotations

import torch
from torch import nn

from methods.ssl.algorithms.flexmatch.flexmatch import (
    FlexMatchAlgorithm,
    compute_flexmatch_step,
)
from methods.ssl.hooks.adaptive_thresholding import FlexMatchThresholdingHook
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


def test_query_ssl_algorithm_registry_builds_flexmatch_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="flexmatch",
        parameters={
            "temperature": 0.5,
            "p_cutoff": 0.95,
            "hard_label": True,
            "thresh_warmup": True,
            "lambda_u": 1.0,
            "supervised_loss_weight": 1.0,
        },
    )

    assert isinstance(algorithm, FlexMatchAlgorithm)
    assert algorithm.algorithm_name == "flexmatch"
    assert algorithm.p_cutoff == 0.95
    assert algorithm.thresh_warmup is True


def test_query_ssl_algorithm_descriptor_exposes_flexmatch_view_spec() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("flexmatch")

    assert descriptor.algorithm_name == "flexmatch"
    assert descriptor.display_name == "FlexMatch"
    assert descriptor.required_views.view_names == ("text", "aug_0", "aug_1")
    assert descriptor.required_views.view_builder_name == "usb_multiview"
    assert descriptor.default_uses_labeled_batches is True


def test_flexmatch_thresholding_hook_matches_usb_initial_warmup_update() -> None:
    hook = FlexMatchThresholdingHook(
        ulb_dest_len=4,
        num_classes=2,
        thresh_warmup=True,
    )
    algorithm = FlexMatchAlgorithm(
        temperature=0.5,
        p_cutoff=0.95,
        hard_label=True,
        thresh_warmup=True,
    )
    probs_x_ulb_w = torch.tensor(
        [[0.98, 0.02], [0.80, 0.20]],
        dtype=torch.float32,
    )

    mask = hook.masking(
        algorithm,
        logits_x_ulb=probs_x_ulb_w,
        idx_ulb=torch.tensor([1, 2], dtype=torch.long),
        softmax_x_ulb=False,
    )

    assert torch.equal(mask, torch.tensor([1.0, 1.0]))
    assert hook.selected_label.tolist() == [-1, 0, -1, -1]
    assert torch.allclose(hook.classwise_acc, torch.tensor([1.0 / 3.0, 0.0]))


def test_compute_flexmatch_step_matches_usb_adaptive_threshold_flow() -> None:
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
        "row_indices": torch.tensor([0, 1], dtype=torch.long),
        "weak_input_ids": torch.ones((2, 2), dtype=torch.long),
        "weak_attention_mask": torch.ones((2, 2), dtype=torch.long),
        "strong_input_ids": torch.ones((2, 2), dtype=torch.long),
        "strong_attention_mask": torch.ones((2, 2), dtype=torch.long),
    }
    algorithm = FlexMatchAlgorithm(
        temperature=0.5,
        p_cutoff=0.95,
        hard_label=True,
        thresh_warmup=True,
        lambda_u=1.0,
        supervised_loss_weight=1.0,
    )
    algorithm.configure_dataset(num_classes=2, unlabeled_row_count=4)
    assert algorithm.masking_hook is not None

    output = compute_flexmatch_step(
        model=model,  # type: ignore[arg-type]
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        temperature=0.5,
        p_cutoff=0.95,
        hard_label=True,
        lambda_u=1.0,
        supervised_loss_weight=1.0,
        masking_hook=algorithm.masking_hook,
        algorithm=algorithm,
    )

    weak_probs = torch.softmax(torch.tensor([[6.0, 0.0], [0.8, 0.0]]), dim=-1)
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
    expected_unsup_loss = strong_losses.mean()

    assert torch.equal(output.debug_tensors["mask"], torch.tensor([1.0, 1.0]))
    assert torch.isclose(output.loss_components["sup_loss"], expected_sup_loss)
    assert torch.isclose(output.loss_components["unsup_loss"], expected_unsup_loss)
    assert torch.isclose(output.total_loss, expected_sup_loss + expected_unsup_loss)
    assert torch.isclose(output.metrics["util_ratio"], torch.tensor(1.0))
    assert torch.allclose(
        output.debug_tensors["classwise_acc"],
        algorithm.masking_hook.classwise_acc,
    )
    assert algorithm.masking_hook.selected_label.tolist() == [0, -1, -1, -1]


def test_flexmatch_algorithm_requires_dataset_state_before_step() -> None:
    algorithm = FlexMatchAlgorithm(
        temperature=0.5,
        p_cutoff=0.95,
    )

    try:
        algorithm.compute_step(
            model=_SequentialLogitModel([]),  # type: ignore[arg-type]
            labeled_batch=None,
            unlabeled_batch={
                "row_indices": torch.tensor([0], dtype=torch.long),
                "weak_input_ids": torch.ones((1, 2), dtype=torch.long),
                "weak_attention_mask": torch.ones((1, 2), dtype=torch.long),
                "strong_input_ids": torch.ones((1, 2), dtype=torch.long),
                "strong_attention_mask": torch.ones((1, 2), dtype=torch.long),
            },
        )
    except ValueError as exc:
        assert "configure_dataset" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("FlexMatch should require dataset state.")

"""Reusable FreeMatch method core tests."""

from __future__ import annotations

import torch
from torch import nn

from methods.ssl.algorithms.freematch.freematch import (
    FreeMatchAlgorithm,
    build_freematch_pseudo_label,
    compute_freematch_step,
)
from methods.ssl.hooks.adaptive_thresholding import FreeMatchThresholdingHook
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


def test_query_ssl_algorithm_registry_builds_freematch_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="freematch",
        parameters={
            "temperature": 0.5,
            "hard_label": True,
            "ema_p": 0.999,
            "ent_loss_ratio": 0.01,
            "use_quantile": False,
            "clip_thresh": False,
            "lambda_u": 1.0,
            "supervised_loss_weight": 1.0,
        },
    )

    assert isinstance(algorithm, FreeMatchAlgorithm)
    assert algorithm.algorithm_name == "freematch"
    assert algorithm.ema_p == 0.999
    assert algorithm.ent_loss_ratio == 0.01


def test_query_ssl_algorithm_descriptor_exposes_freematch_view_spec() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("freematch")

    assert descriptor.algorithm_name == "freematch"
    assert descriptor.display_name == "FreeMatch"
    assert descriptor.required_views.view_names == ("text", "aug_0", "aug_1")
    assert descriptor.required_views.view_builder_name == "usb_multiview"
    assert descriptor.default_uses_labeled_batches is True


def test_freematch_thresholding_hook_matches_usb_ema_update() -> None:
    hook = FreeMatchThresholdingHook(
        num_classes=2,
        momentum=0.5,
    )
    algorithm = FreeMatchAlgorithm(
        temperature=0.5,
        ema_p=0.5,
        use_quantile=False,
        clip_thresh=False,
    )
    probs_x_ulb_w = torch.tensor(
        [[0.8, 0.2], [0.4, 0.6]],
        dtype=torch.float32,
    )

    mask = hook.masking(
        algorithm,
        logits_x_ulb=probs_x_ulb_w,
        softmax_x_ulb=False,
    )

    assert torch.equal(mask, torch.tensor([1.0, 1.0]))
    assert torch.allclose(hook.time_p, torch.tensor(0.6))
    assert torch.allclose(hook.p_model, torch.tensor([0.55, 0.45]))
    assert torch.allclose(hook.label_hist, torch.tensor([0.5, 0.5]))
    assert algorithm.p_model is hook.p_model
    assert algorithm.label_hist is hook.label_hist
    assert algorithm.time_p is hook.time_p


def test_build_freematch_soft_pseudo_label_matches_usb_temperature_path() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    soft_targets = build_freematch_pseudo_label(
        logits_x_ulb_w=logits,
        hard_label=False,
        temperature=0.5,
    )
    hard_targets = build_freematch_pseudo_label(
        logits_x_ulb_w=logits,
        hard_label=True,
        temperature=0.5,
    )

    assert torch.allclose(soft_targets, torch.softmax(logits / 0.5, dim=-1))
    assert hard_targets.tolist() == [0, 1]


def test_compute_freematch_step_matches_usb_adaptive_threshold_flow() -> None:
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
    algorithm = FreeMatchAlgorithm(
        temperature=0.5,
        hard_label=True,
        ema_p=0.999,
        ent_loss_ratio=0.01,
        use_quantile=False,
        clip_thresh=False,
        lambda_u=1.0,
        supervised_loss_weight=1.0,
    )
    algorithm.configure_dataset(num_classes=2, unlabeled_row_count=4)
    assert algorithm.masking_hook is not None

    output = compute_freematch_step(
        model=model,  # type: ignore[arg-type]
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        temperature=0.5,
        hard_label=True,
        lambda_u=1.0,
        ent_loss_ratio=0.01,
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
    assert output.loss_components["ent_loss"].ndim == 0
    assert torch.isclose(
        output.total_loss,
        expected_sup_loss
        + expected_unsup_loss
        + 0.01 * output.loss_components["ent_loss"],
    )
    assert torch.isclose(output.metrics["util_ratio"], torch.tensor(1.0))
    assert output.metrics["time_p"] is algorithm.masking_hook.time_p
    assert torch.allclose(
        output.debug_tensors["p_model"],
        algorithm.masking_hook.p_model,
    )
    assert torch.allclose(
        output.debug_tensors["label_hist"],
        algorithm.masking_hook.label_hist,
    )


def test_freematch_algorithm_requires_dataset_state_before_step() -> None:
    algorithm = FreeMatchAlgorithm(
        temperature=0.5,
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
        raise AssertionError("FreeMatch should require dataset state.")

"""Reusable Dash method core tests."""

from __future__ import annotations

import torch
from torch import nn

from methods.ssl.algorithms.dash.dash import (
    DashAlgorithm,
    DashThresholdingHook,
    compute_dash_step,
)
from methods.ssl.base import QuerySslStepContext
from methods.ssl.registry import (
    build_query_ssl_algorithm,
    resolve_query_ssl_algorithm_descriptor,
)


class _TokenSumClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))

    def forward(self, *, input_ids, attention_mask):
        values = (input_ids.float() * attention_mask.float()).sum(dim=1)
        features = torch.stack([values, -values], dim=1)
        return features @ self.weight


def _labeled_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }


def _unlabeled_batch() -> dict[str, torch.Tensor]:
    return {
        "weak_input_ids": torch.tensor([[1, 1], [0, 1]], dtype=torch.long),
        "weak_attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        "strong_input_ids": torch.tensor([[1, 0], [1, 1]], dtype=torch.long),
        "strong_attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
    }


def _step_context(*, epoch_index: int = 1, step_index: int = 1) -> QuerySslStepContext:
    return QuerySslStepContext(
        epoch_index=epoch_index,
        step_index=step_index,
        global_step=(epoch_index - 1) * 2 + step_index,
        total_train_steps=40,
        num_classes=2,
        device=torch.device("cpu"),
    )


def test_query_ssl_algorithm_registry_builds_dash_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="dash",
        parameters={
            "T": 0.5,
            "gamma": 1.27,
            "C": 1.0001,
            "rho_min": 0.05,
            "num_wu_iter": 256,
            "lambda_u": 1.0,
            "rho_init": 0.7,
        },
    )

    assert isinstance(algorithm, DashAlgorithm)
    assert algorithm.algorithm_name == "dash"
    assert algorithm.T == 0.5
    assert algorithm.num_wu_iter == 256
    assert algorithm.needs_initial_selection_loss is False
    assert algorithm.initial_selection_warmup_steps == 0


def test_query_ssl_algorithm_descriptor_exposes_dash_view_spec() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("dash")

    assert descriptor.algorithm_name == "dash"
    assert descriptor.display_name == "Dash"
    assert descriptor.required_views.view_names == ("text", "aug_0", "aug_1")
    assert descriptor.required_views.view_builder_name == "usb_multiview"
    assert descriptor.default_uses_labeled_batches is True
    assert descriptor.runtime_requirements.step_context_required is True


def test_dash_threshold_hook_matches_usb_dynamic_threshold_schedule() -> None:
    hook = DashThresholdingHook(rho_min=0.05, gamma=2.0, C=1.0)
    hook.configure_initial_selection_loss(selection_loss=0.8)

    mask = hook.build_mask(
        logits_x_ulb_w=torch.tensor([[4.0, -4.0], [0.2, -0.2]]),
        temperature=0.5,
        step_context=_step_context(epoch_index=1, step_index=1),
    )
    assert hook.state.rho == 0.8
    assert hook.state.rho_update_cnt == 1
    assert mask.tolist() == [1.0, 1.0]

    hook.build_mask(
        logits_x_ulb_w=torch.tensor([[4.0, -4.0], [0.2, -0.2]]),
        temperature=0.5,
        step_context=_step_context(epoch_index=11, step_index=1),
    )
    assert hook.state.rho == 0.4
    assert hook.state.rho_update_cnt == 2


def test_compute_dash_step_uses_weak_ce_mask_and_student_strong_ce() -> None:
    model = _TokenSumClassifier()
    hook = DashThresholdingHook(rho_min=0.05, gamma=1.27, C=1.0001)
    hook.configure_initial_selection_loss(selection_loss=1.0)

    output = compute_dash_step(
        model=model,
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
        step_context=_step_context(),
        temperature=0.5,
        lambda_u=1.0,
        thresholding_hook=hook,
    )

    with torch.no_grad():
        labeled_logits = model(
            input_ids=_labeled_batch()["input_ids"],
            attention_mask=_labeled_batch()["attention_mask"],
        )
        weak_logits = model(
            input_ids=_unlabeled_batch()["weak_input_ids"],
            attention_mask=_unlabeled_batch()["weak_attention_mask"],
        )
        strong_logits = model(
            input_ids=_unlabeled_batch()["strong_input_ids"],
            attention_mask=_unlabeled_batch()["strong_attention_mask"],
        )
    soft_targets = torch.softmax(weak_logits.detach() / 0.5, dim=-1)
    weak_ce = -(soft_targets * torch.log_softmax(weak_logits, dim=-1)).sum(dim=-1)
    expected_mask = weak_ce.le(hook.state.rho).to(weak_logits.dtype)
    expected_unsup = (
        -(soft_targets * torch.log_softmax(strong_logits, dim=-1)).sum(dim=-1)
        * expected_mask
    ).mean()
    expected_sup = nn.functional.cross_entropy(
        labeled_logits,
        _labeled_batch()["labels"],
        reduction="mean",
    )

    assert torch.allclose(output.debug_tensors["mask"], expected_mask)
    assert torch.isclose(output.loss_components["sup_loss"], expected_sup)
    assert torch.isclose(output.loss_components["unsup_loss"], expected_unsup)
    assert torch.isclose(output.total_loss, expected_sup + expected_unsup)


def test_dash_algorithm_state_roundtrips_threshold_state() -> None:
    algorithm = DashAlgorithm(
        T=0.5,
        gamma=2.0,
        C=1.0,
        rho_min=0.05,
        num_wu_iter=1024,
    )
    algorithm.configure_initial_selection_loss(selection_loss=0.8)
    algorithm.compute_step_with_context(
        model=_TokenSumClassifier(),
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
        step_context=_step_context(epoch_index=1, step_index=1),
    )
    state = algorithm.export_state()

    restored = DashAlgorithm(T=0.5, gamma=2.0, C=1.0, rho_min=0.05)
    restored.load_state(state)
    restored_state = restored.export_state()

    assert restored_state["rho_init"] == state["rho_init"]
    assert restored_state["rho"] == state["rho"]
    assert restored_state["rho_update_cnt"] == state["rho_update_cnt"]
    assert state["num_wu_iter"] == 1024
    assert state["rho_init_source"] == "supervised_warmup_selection_loss"

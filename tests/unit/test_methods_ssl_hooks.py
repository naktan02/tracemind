"""Reusable SSL hook tests."""

from __future__ import annotations

from methods.ssl.hooks.consistency import CrossEntropyConsistencyLossHook
from methods.ssl.hooks.distribution_alignment import QueueDistributionAlignmentHook
from methods.ssl.hooks.masking import FixedThresholdMaskingHook
from methods.ssl.hooks.pseudo_labeling import (
    HardOrSoftPseudoLabelingHook,
    PseudoLabelingConfig,
)


def test_pseudo_labeling_hook_builds_hard_and_soft_targets() -> None:
    import torch

    probs = torch.tensor([[0.8, 0.2], [0.4, 0.6]], dtype=torch.float32)
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    hook = HardOrSoftPseudoLabelingHook()

    hard_targets = hook.generate_targets(
        probs_x_ulb_w=probs,
        config=PseudoLabelingConfig(use_hard_label=True, temperature=0.5),
    )
    soft_targets = hook.generate_targets(
        probs_x_ulb_w=probs,
        config=PseudoLabelingConfig(use_hard_label=False, temperature=0.5),
    )
    soft_targets_from_logits = hook.generate_targets_from_logits(
        logits_x_ulb_w=logits,
        config=PseudoLabelingConfig(use_hard_label=False, temperature=0.5),
    )

    assert hook.hook_name == "hard_or_soft_pseudo_labeling"
    assert hard_targets.tolist() == [0, 1]
    assert torch.equal(soft_targets, probs)
    assert torch.allclose(
        soft_targets_from_logits,
        torch.softmax(logits / 0.5, dim=-1),
    )


def test_fixed_threshold_masking_hook_builds_usb_style_mask() -> None:
    import torch

    probs = torch.tensor([[0.96, 0.04], [0.60, 0.40]], dtype=torch.float32)
    hook = FixedThresholdMaskingHook()

    mask = hook.build_mask(probs_x_ulb_w=probs, p_cutoff=0.95)

    assert hook.hook_name == "fixed_threshold"
    assert torch.equal(mask, torch.tensor([1.0, 0.0], dtype=torch.float32))


def test_cross_entropy_consistency_hook_applies_masked_ce() -> None:
    import torch

    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32)
    targets = torch.tensor([0, 0], dtype=torch.long)
    mask = torch.tensor([1.0, 0.0], dtype=torch.float32)
    hook = CrossEntropyConsistencyLossHook()

    loss = hook.compute_loss(logits=logits, targets=targets, mask=mask)
    expected = torch.nn.functional.cross_entropy(
        logits,
        targets,
        reduction="none",
    )

    assert hook.hook_name == "cross_entropy_consistency"
    assert torch.isclose(loss, (expected * mask).mean())


def test_queue_distribution_alignment_updates_model_queue() -> None:
    import torch

    hook = QueueDistributionAlignmentHook(num_classes=2, queue_length=2)
    probs = torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float32)

    aligned = hook.dist_align(probs_x_ulb=probs)

    expected_model_queue = torch.tensor(
        [[0.7, 0.3], [0.0, 0.0]],
        dtype=torch.float32,
    )
    expected = (
        probs
        * (torch.tensor([0.5, 0.5]) + 1e-6)
        / (expected_model_queue.mean(dim=0) + 1e-6)
    )
    expected = expected / expected.sum(dim=-1, keepdim=True)

    assert hook.hook_name == "queue_distribution_alignment"
    assert torch.allclose(hook.p_model, expected_model_queue)
    assert hook.p_model_ptr.tolist() == [1]
    assert torch.allclose(aligned, expected)


def test_queue_distribution_alignment_roundtrips_state() -> None:
    import torch

    hook = QueueDistributionAlignmentHook(
        num_classes=2,
        queue_length=2,
        p_target_type="model",
    )
    hook.dist_align(
        probs_x_ulb=torch.tensor([[0.8, 0.2]], dtype=torch.float32),
        probs_x_lb=torch.tensor([[0.1, 0.9]], dtype=torch.float32),
    )
    state = hook.export_state()

    restored = QueueDistributionAlignmentHook(
        num_classes=2,
        queue_length=2,
        p_target_type="model",
    )
    restored.load_state(state, device=torch.device("cpu"))

    assert torch.allclose(restored.p_model, hook.p_model)
    assert torch.equal(restored.p_model_ptr, hook.p_model_ptr)
    assert torch.allclose(restored.p_target, hook.p_target)
    assert restored.p_target_ptr is not None
    assert hook.p_target_ptr is not None
    assert torch.equal(restored.p_target_ptr, hook.p_target_ptr)

"""Reusable SSL hook tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from methods.ssl.hooks.consistency import CrossEntropyConsistencyLossHook
from methods.ssl.hooks.masking import FixedThresholdMaskingHook
from methods.ssl.hooks.pseudo_labeling import (
    HardOrSoftPseudoLabelingHook,
    PseudoLabelingConfig,
)
from methods.ssl.hooks.registry import (
    build_pseudo_label_selection_hook,
)
from methods.ssl.hooks.selection import PseudoLabelSelectionConfig
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PSEUDO_LABEL_EVIDENCE_V1,
    PseudoLabelEvidence,
)


def _build_evidence() -> PseudoLabelEvidence:
    return PseudoLabelEvidence(
        schema_version=PSEUDO_LABEL_EVIDENCE_V1,
        evidence_id="evidence:q1",
        source_event_ref="q1",
        occurred_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        label="anxiety",
        confidence=0.62,
        confidence_kind="prototype_similarity",
        margin=0.01,
        top1_label="anxiety",
        top1_score=0.62,
        top2_label="depression",
        top2_score=0.61,
        raw_scores={"anxiety": 0.62, "depression": 0.61, "normal": 0.1},
    )


def test_pseudo_labeling_hook_builds_hard_and_soft_targets() -> None:
    import torch

    probs = torch.tensor([[0.8, 0.2], [0.4, 0.6]], dtype=torch.float32)
    hook = HardOrSoftPseudoLabelingHook()

    hard_targets = hook.generate_targets(
        probs_x_ulb_w=probs,
        config=PseudoLabelingConfig(use_hard_label=True, temperature=0.5),
    )
    soft_targets = hook.generate_targets(
        probs_x_ulb_w=probs,
        config=PseudoLabelingConfig(use_hard_label=False, temperature=0.5),
    )

    assert hook.hook_name == "hard_or_soft_pseudo_labeling"
    assert hard_targets.tolist() == [0, 1]
    assert torch.equal(soft_targets, probs)


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


def test_margin_threshold_selection_hook_requires_margin_cutoff() -> None:
    selection_hook = build_pseudo_label_selection_hook("top1_margin_threshold")

    decision = selection_hook.evaluate(
        evidence=_build_evidence(),
        config=PseudoLabelSelectionConfig(
            confidence_threshold=0.6,
            margin_threshold=0.02,
        ),
    )

    assert selection_hook.hook_name == "top1_margin_threshold"
    assert decision.accepted is False
    assert decision.confidence == pytest.approx(0.62)
    assert decision.margin == pytest.approx(0.01)


def test_fixed_confidence_selection_hook_ignores_margin_cutoff() -> None:
    selection_hook = build_pseudo_label_selection_hook("top1_confidence_only")

    decision = selection_hook.evaluate(
        evidence=_build_evidence(),
        config=PseudoLabelSelectionConfig(
            confidence_threshold=0.6,
            margin_threshold=0.99,
        ),
    )

    assert decision.accepted is True
    assert decision.label == "anxiety"
    assert decision.runner_up_label == "depression"

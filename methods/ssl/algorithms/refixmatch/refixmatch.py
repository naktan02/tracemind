"""USB ReFixMatch core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ...base import QuerySslStepResult, TextBatchClassifier
from ...common import compute_prob
from ...hooks.consistency import CrossEntropyConsistencyLossHook
from ...hooks.masking import FixedThresholdMaskingHook
from ...hooks.objective import SslObjectiveHooks
from ...hooks.pseudo_labeling import (
    HardOrSoftPseudoLabelingHook,
    PseudoLabelingConfig,
)
from ...hooks.supervised import compute_labeled_cross_entropy_loss
from ...registry import register_query_ssl_algorithm
from ..usb_consistency import (
    USB_MULTIVIEW_REQUIRED_VIEWS,
    compute_unlabeled_weak_strong_logits,
    validate_usb_consistency_loaders,
)

_REFIX_KL_TEMPERATURE = 0.5


def build_refixmatch_objective_hooks() -> SslObjectiveHooks:
    """ReFixMatch의 FixMatch-compatible hook 조합을 만든다."""

    return SslObjectiveHooks(
        pseudo_labeling=HardOrSoftPseudoLabelingHook(),
        masking=FixedThresholdMaskingHook(),
        consistency_loss=CrossEntropyConsistencyLossHook(),
    )


class ReFixMatchAlgorithm:
    """ReFixMatch를 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "refixmatch"

    def __init__(
        self,
        *,
        temperature: float,
        p_cutoff: float,
        hard_label: bool = True,
        lambda_u: float = 1.0,
        supervised_loss_weight: float = 1.0,
        hooks: SslObjectiveHooks | None = None,
    ) -> None:
        self.temperature = float(temperature)
        self.p_cutoff = float(p_cutoff)
        self.hard_label = bool(hard_label)
        self.lambda_u = float(lambda_u)
        self.supervised_loss_weight = float(supervised_loss_weight)
        self.hooks = hooks or build_refixmatch_objective_hooks()

    @property
    def uses_labeled_batches(self) -> bool:
        return self.supervised_loss_weight > 0

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        validate_usb_consistency_loaders(
            algorithm_name="ReFixMatch",
            train_loader_length=train_loader_length,
            unlabeled_loader_length=unlabeled_loader_length,
            supervised_loss_weight=self.supervised_loss_weight,
        )

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Tensor],
    ) -> QuerySslStepResult:
        return compute_refixmatch_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            temperature=self.temperature,
            p_cutoff=self.p_cutoff,
            hard_label=self.hard_label,
            lambda_u=self.lambda_u,
            supervised_loss_weight=self.supervised_loss_weight,
            hooks=self.hooks,
        )


def compute_refixmatch_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Tensor],
    temperature: float,
    p_cutoff: float,
    hard_label: bool = True,
    lambda_u: float = 1.0,
    supervised_loss_weight: float = 1.0,
    hooks: SslObjectiveHooks | None = None,
) -> QuerySslStepResult:
    """USB `semilearn/algorithms/refixmatch/refixmatch.py::train_step` 핵심."""

    sup_loss = compute_labeled_cross_entropy_loss(
        model=model,
        labeled_batch=labeled_batch,
    )
    logits_x_ulb_s, logits_x_ulb_w = compute_unlabeled_weak_strong_logits(
        model=model,
        unlabeled_batch=unlabeled_batch,
    )
    if sup_loss is None:
        sup_loss = logits_x_ulb_s.new_zeros(())

    probs_x_ulb_w = compute_prob(logits_x_ulb_w.detach())
    effective_hooks = hooks or build_refixmatch_objective_hooks()
    mask = effective_hooks.masking.build_mask(
        probs_x_ulb_w=probs_x_ulb_w,
        p_cutoff=p_cutoff,
    )
    pseudo_label = effective_hooks.pseudo_labeling.generate_targets(
        probs_x_ulb_w=probs_x_ulb_w,
        config=PseudoLabelingConfig(
            use_hard_label=hard_label,
            temperature=temperature,
        ),
    )
    unsup_loss = effective_hooks.consistency_loss.compute_loss(
        logits=logits_x_ulb_s,
        targets=pseudo_label,
        mask=mask,
    )
    refix_loss = refixmatch_low_confidence_kl_loss(
        logits=logits_x_ulb_s,
        targets=probs_x_ulb_w,
        mask=mask,
    )
    total_loss = (
        float(supervised_loss_weight) * sup_loss
        + float(lambda_u) * unsup_loss
        + float(lambda_u) * refix_loss
    )
    return QuerySslStepResult(
        total_loss=total_loss,
        loss_components={
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "refix_loss": refix_loss,
        },
        metrics={"util_ratio": mask.float().mean()},
        debug_tensors={
            "mask": mask,
            "probs_x_ulb_w": probs_x_ulb_w,
        },
    )


def refixmatch_low_confidence_kl_loss(
    *,
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> Tensor:
    """USB ConsistencyLoss(name='kl')의 ReFixMatch 사용 형태."""

    loss = F.kl_div(
        F.log_softmax(logits / _REFIX_KL_TEMPERATURE, dim=-1),
        F.softmax(targets / _REFIX_KL_TEMPERATURE, dim=-1),
        reduction="none",
    )
    complement = (1.0 - mask).unsqueeze(dim=-1).repeat(1, logits.shape[1])
    return torch.sum(loss * complement, dim=1).mean()


@register_query_ssl_algorithm(
    "refixmatch",
    "re_fixmatch",
    display_name="ReFixMatch",
    required_views=USB_MULTIVIEW_REQUIRED_VIEWS,
    default_uses_labeled_batches=True,
)
def build_refixmatch_algorithm(parameters: Mapping[str, Any]) -> ReFixMatchAlgorithm:
    """Hydra method parameter mapping으로 ReFixMatch algorithm을 만든다."""

    return ReFixMatchAlgorithm(
        temperature=float(parameters["temperature"]),
        p_cutoff=float(parameters["p_cutoff"]),
        hard_label=bool(parameters.get("hard_label", True)),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
        supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
    )

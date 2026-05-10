"""USB FixMatch core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ...base import QuerySslRequiredViews, QuerySslStepOutput, TextBatchClassifier
from ...common import (
    compute_prob,
)
from ...hooks.consistency import CrossEntropyConsistencyLossHook
from ...hooks.masking import FixedThresholdMaskingHook
from ...hooks.objective import SslObjectiveHooks
from ...hooks.pseudo_labeling import (
    HardOrSoftPseudoLabelingHook,
    PseudoLabelingConfig,
)
from ...registry import register_query_ssl_algorithm


class FixMatchConfig:
    """기존 FixMatch wrapper가 넘기는 핵심 하이퍼파라미터 묶음."""

    def __init__(
        self,
        *,
        temperature: float,
        p_cutoff: float,
        hard_label: bool = True,
        lambda_u: float = 1.0,
        supervised_loss_weight: float = 1.0,
    ) -> None:
        self.temperature = float(temperature)
        self.p_cutoff = float(p_cutoff)
        self.hard_label = bool(hard_label)
        self.lambda_u = float(lambda_u)
        self.supervised_loss_weight = float(supervised_loss_weight)


class FixMatchStepOutput:
    """FixMatch 한 step의 loss/diagnostics 결과."""

    def __init__(
        self,
        *,
        total_loss: Tensor,
        sup_loss: Tensor,
        unsup_loss: Tensor,
        mask: Tensor,
    ) -> None:
        self.total_loss = total_loss
        self.sup_loss = sup_loss
        self.unsup_loss = unsup_loss
        self.mask = mask

    @property
    def util_ratio(self) -> Tensor:
        return self.mask.float().mean()

    @property
    def loss_components(self) -> dict[str, Tensor]:
        return {
            "sup_loss": self.sup_loss,
            "unsup_loss": self.unsup_loss,
        }

    @property
    def metrics(self) -> dict[str, Tensor]:
        return {"util_ratio": self.util_ratio}


def build_fixmatch_objective_hooks() -> SslObjectiveHooks:
    """FixMatch baseline의 tensor-level hook 조합을 만든다."""

    return SslObjectiveHooks(
        pseudo_labeling=HardOrSoftPseudoLabelingHook(),
        masking=FixedThresholdMaskingHook(),
        consistency_loss=CrossEntropyConsistencyLossHook(),
    )


class FixMatchAlgorithm:
    """FixMatch를 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "fixmatch"

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
        self.hooks = hooks or build_fixmatch_objective_hooks()

    @property
    def uses_labeled_batches(self) -> bool:
        return self.supervised_loss_weight > 0

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        if unlabeled_loader_length == 0:
            raise ValueError("FixMatch unlabeled_loader must not be empty.")
        if self.supervised_loss_weight > 0 and train_loader_length == 0:
            raise ValueError(
                "FixMatch labeled train_loader must not be empty when "
                "supervised_loss_weight > 0."
            )

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Tensor],
    ) -> QuerySslStepOutput:
        return compute_fixmatch_step(
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


def compute_fixmatch_step(
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
) -> FixMatchStepOutput:
    """USB `semilearn/algorithms/fixmatch/fixmatch.py::train_step` 핵심."""

    if labeled_batch is None:
        sup_loss = None
    else:
        logits_x_lb = model(
            input_ids=labeled_batch["input_ids"],
            attention_mask=labeled_batch["attention_mask"],
        )
        sup_loss = F.cross_entropy(
            logits_x_lb, labeled_batch["labels"], reduction="mean"
        )

    logits_x_ulb_s = model(
        input_ids=unlabeled_batch["strong_input_ids"],
        attention_mask=unlabeled_batch["strong_attention_mask"],
    )
    with torch.no_grad():
        logits_x_ulb_w = model(
            input_ids=unlabeled_batch["weak_input_ids"],
            attention_mask=unlabeled_batch["weak_attention_mask"],
        )
    if sup_loss is None:
        sup_loss = logits_x_ulb_s.new_zeros(())

    probs_x_ulb_w = compute_prob(logits_x_ulb_w.detach())
    effective_hooks = hooks or build_fixmatch_objective_hooks()
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
    total_loss = supervised_loss_weight * sup_loss + lambda_u * unsup_loss
    return FixMatchStepOutput(
        total_loss=total_loss,
        sup_loss=sup_loss,
        unsup_loss=unsup_loss,
        mask=mask,
    )


@register_query_ssl_algorithm(
    "fixmatch",
    display_name="FixMatch",
    required_views=QuerySslRequiredViews(
        view_names=("text", "aug_0", "aug_1"),
        view_builder_name="usb_multiview",
    ),
    default_uses_labeled_batches=True,
)
def build_fixmatch_algorithm(parameters: Mapping[str, Any]) -> FixMatchAlgorithm:
    """Hydra method parameter mapping으로 FixMatch algorithm을 만든다."""

    return FixMatchAlgorithm(
        temperature=float(parameters["temperature"]),
        p_cutoff=float(parameters["p_cutoff"]),
        hard_label=bool(parameters.get("hard_label", True)),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
        supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
    )

"""USB AdaMatch core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ...base import QuerySslStepResult, TextBatchClassifier
from ...common import compute_prob
from ...hooks.consistency import ConsistencyLossHook, CrossEntropyConsistencyLossHook
from ...hooks.pseudo_labeling import (
    HardOrSoftPseudoLabelingHook,
    PseudoLabelingConfig,
    PseudoLabelingHook,
)
from ...registry import register_query_ssl_algorithm
from ..usb_consistency import (
    USB_MULTIVIEW_REQUIRED_VIEWS,
    compute_unlabeled_weak_strong_logits,
)


class AdaMatchDistAlignHook:
    """USB `DistAlignEMAHook(..., p_target_type='model')`의 EMA state."""

    hook_name: str = "adamatch_dist_align_ema"

    def __init__(
        self,
        *,
        num_classes: int,
        momentum: float = 0.999,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = int(num_classes)
        self.m = float(momentum)
        self.p_model: Tensor | None = None
        self.p_target = torch.ones((self.num_classes,)) / self.num_classes

    @torch.no_grad()
    def dist_align(
        self,
        *,
        probs_x_ulb: Tensor,
        probs_x_lb: Tensor,
        algorithm: AdaMatchAlgorithm | None = None,
    ) -> Tensor:
        """USB AdaMatch distribution alignment를 적용한 unlabeled probability."""

        self._move_state_to_device(probs_x_ulb.device)
        self.update_p(probs_x_ulb=probs_x_ulb, probs_x_lb=probs_x_lb)

        if self.p_model is None:  # pragma: no cover - defensive
            raise RuntimeError("AdaMatch distribution alignment state was not updated.")

        aligned_probs = probs_x_ulb * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        aligned_probs = aligned_probs / aligned_probs.sum(dim=-1, keepdim=True)

        if algorithm is not None:
            algorithm.p_model = self.p_model
            algorithm.p_target = self.p_target
        return aligned_probs

    @torch.no_grad()
    def update_p(
        self,
        *,
        probs_x_ulb: Tensor,
        probs_x_lb: Tensor,
    ) -> None:
        """USB `DistAlignEMAHook.update_p`의 non-distributed EMA update."""

        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model is None:
            self.p_model = probs_x_ulb.mean(dim=0)
        else:
            self.p_model = self.p_model * self.m + probs_x_ulb.mean(dim=0) * (
                1 - self.m
            )
        self.p_target = self.p_target * self.m + probs_x_lb.mean(dim=0) * (
            1 - self.m
        )

    def _move_state_to_device(self, device: torch.device) -> None:
        if self.p_target.device != device:
            self.p_target = self.p_target.to(device)
        if self.p_model is not None and self.p_model.device != device:
            self.p_model = self.p_model.to(device)


class AdaMatchThresholdingHook:
    """USB `AdaMatchThresholdingHook`의 relative confidence threshold."""

    hook_name: str = "adamatch_relative_confidence_threshold"

    @torch.no_grad()
    def masking(
        self,
        algorithm: AdaMatchAlgorithm,
        *,
        probs_x_lb: Tensor,
        probs_x_ulb: Tensor,
    ) -> Tensor:
        max_probs_lb, _ = probs_x_lb.detach().max(dim=-1)
        p_cutoff = max_probs_lb.mean() * algorithm.p_cutoff
        max_probs_ulb, _ = probs_x_ulb.detach().max(dim=-1)
        return max_probs_ulb.ge(p_cutoff).to(max_probs_ulb.dtype)


class AdaMatchAlgorithm:
    """AdaMatch를 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "adamatch"

    def __init__(
        self,
        *,
        temperature: float,
        p_cutoff: float,
        hard_label: bool = True,
        ema_p: float = 0.999,
        lambda_u: float = 1.0,
        supervised_loss_weight: float = 1.0,
        pseudo_labeling_hook: PseudoLabelingHook | None = None,
        consistency_loss_hook: ConsistencyLossHook | None = None,
        masking_hook: AdaMatchThresholdingHook | None = None,
    ) -> None:
        self.temperature = float(temperature)
        self.p_cutoff = float(p_cutoff)
        self.hard_label = bool(hard_label)
        self.ema_p = float(ema_p)
        self.lambda_u = float(lambda_u)
        self.supervised_loss_weight = float(supervised_loss_weight)
        self.pseudo_labeling_hook = (
            pseudo_labeling_hook or HardOrSoftPseudoLabelingHook()
        )
        self.consistency_loss_hook = (
            consistency_loss_hook or CrossEntropyConsistencyLossHook()
        )
        self.masking_hook = masking_hook or AdaMatchThresholdingHook()
        self.dist_align_hook: AdaMatchDistAlignHook | None = None
        self.p_model: Tensor | None = None
        self.p_target: Tensor | None = None

    @property
    def uses_labeled_batches(self) -> bool:
        return True

    def configure_dataset(
        self,
        *,
        num_classes: int,
        unlabeled_row_count: int,
    ) -> None:
        """USB `num_classes` 기반 distribution alignment state를 초기화한다."""

        del unlabeled_row_count
        self.dist_align_hook = AdaMatchDistAlignHook(
            num_classes=num_classes,
            momentum=self.ema_p,
        )
        self.p_model = self.dist_align_hook.p_model
        self.p_target = self.dist_align_hook.p_target

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        if unlabeled_loader_length == 0:
            raise ValueError("AdaMatch unlabeled_loader must not be empty.")
        if train_loader_length == 0:
            raise ValueError("AdaMatch labeled train_loader must not be empty.")

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepResult:
        if self.dist_align_hook is None:
            raise ValueError(
                "AdaMatch requires configure_dataset before compute_step."
            )
        return compute_adamatch_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            temperature=self.temperature,
            p_cutoff=self.p_cutoff,
            hard_label=self.hard_label,
            lambda_u=self.lambda_u,
            supervised_loss_weight=self.supervised_loss_weight,
            pseudo_labeling_hook=self.pseudo_labeling_hook,
            consistency_loss_hook=self.consistency_loss_hook,
            masking_hook=self.masking_hook,
            dist_align_hook=self.dist_align_hook,
            algorithm=self,
        )


def compute_adamatch_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Any],
    temperature: float,
    p_cutoff: float,
    hard_label: bool = True,
    lambda_u: float = 1.0,
    supervised_loss_weight: float = 1.0,
    pseudo_labeling_hook: PseudoLabelingHook | None = None,
    consistency_loss_hook: ConsistencyLossHook | None = None,
    masking_hook: AdaMatchThresholdingHook | None = None,
    dist_align_hook: AdaMatchDistAlignHook,
    algorithm: AdaMatchAlgorithm | None = None,
) -> QuerySslStepResult:
    """USB `semilearn/algorithms/adamatch/adamatch.py::train_step` 핵심."""

    if labeled_batch is None:
        raise ValueError("AdaMatch requires a labeled_batch for dist alignment.")

    logits_x_lb = model(
        input_ids=labeled_batch["input_ids"],
        attention_mask=labeled_batch["attention_mask"],
    )
    sup_loss = F.cross_entropy(logits_x_lb, labeled_batch["labels"], reduction="mean")
    logits_x_ulb_s, logits_x_ulb_w = compute_unlabeled_weak_strong_logits(
        model=model,
        unlabeled_batch=unlabeled_batch,
    )

    probs_x_lb = compute_prob(logits_x_lb.detach())
    probs_x_ulb_w = compute_prob(logits_x_ulb_w.detach())
    probs_x_ulb_w = dist_align_hook.dist_align(
        probs_x_ulb=probs_x_ulb_w,
        probs_x_lb=probs_x_lb,
        algorithm=algorithm,
    )

    masking_algorithm = algorithm or _AdaMatchMaskingAlgorithm(p_cutoff=p_cutoff)
    effective_masking_hook = masking_hook or AdaMatchThresholdingHook()
    mask = effective_masking_hook.masking(
        masking_algorithm,
        probs_x_lb=probs_x_lb,
        probs_x_ulb=probs_x_ulb_w,
    )
    effective_pseudo_labeling_hook = (
        pseudo_labeling_hook or HardOrSoftPseudoLabelingHook()
    )
    pseudo_label = effective_pseudo_labeling_hook.generate_targets(
        probs_x_ulb_w=probs_x_ulb_w,
        config=PseudoLabelingConfig(
            use_hard_label=hard_label,
            temperature=temperature,
        ),
    )
    effective_consistency_loss_hook = (
        consistency_loss_hook or CrossEntropyConsistencyLossHook()
    )
    unsup_loss = effective_consistency_loss_hook.compute_loss(
        logits=logits_x_ulb_s,
        targets=pseudo_label,
        mask=mask,
    )
    total_loss = supervised_loss_weight * sup_loss + lambda_u * unsup_loss

    if dist_align_hook.p_model is None:  # pragma: no cover - defensive
        raise RuntimeError("AdaMatch p_model was not initialized.")

    return QuerySslStepResult(
        total_loss=total_loss,
        loss_components={
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
        },
        metrics={"util_ratio": mask.float().mean()},
        debug_tensors={
            "mask": mask,
            "p_model": dist_align_hook.p_model,
            "p_target": dist_align_hook.p_target,
        },
    )


class _AdaMatchMaskingAlgorithm:
    def __init__(self, *, p_cutoff: float) -> None:
        self.p_cutoff = p_cutoff


@register_query_ssl_algorithm(
    "adamatch",
    display_name="AdaMatch",
    required_views=USB_MULTIVIEW_REQUIRED_VIEWS,
    default_uses_labeled_batches=True,
)
def build_adamatch_algorithm(parameters: Mapping[str, Any]) -> AdaMatchAlgorithm:
    """Hydra method parameter mapping으로 AdaMatch algorithm을 만든다."""

    return AdaMatchAlgorithm(
        temperature=float(parameters["temperature"]),
        p_cutoff=float(parameters["p_cutoff"]),
        hard_label=bool(parameters.get("hard_label", True)),
        ema_p=float(parameters.get("ema_p", 0.999)),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
        supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
    )

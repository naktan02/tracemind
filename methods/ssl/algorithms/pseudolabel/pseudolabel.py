"""USB PseudoLabel core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor
from torch.nn import functional as F

from ...base import QuerySslRequiredViews, QuerySslStepOutput, TextBatchClassifier
from ...common import compute_prob
from ...hooks.consistency import CrossEntropyConsistencyLossHook
from ...hooks.masking import FixedThresholdMaskingHook
from ...hooks.objective import SslObjectiveHooks
from ...hooks.pseudo_labeling import (
    HardOrSoftPseudoLabelingHook,
    PseudoLabelingConfig,
)
from ...registry import register_query_ssl_algorithm


@dataclass(frozen=True, slots=True)
class PseudoLabelConfig:
    """USB PseudoLabel 핵심 하이퍼파라미터 묶음."""

    p_cutoff: float
    unsup_warm_up: float = 0.4
    lambda_u: float = 1.0
    supervised_loss_weight: float = 1.0


@dataclass(slots=True)
class PseudoLabelStepOutput:
    """PseudoLabel 한 step의 loss/diagnostics 결과."""

    total_loss: Tensor
    sup_loss: Tensor
    unsup_loss: Tensor
    mask: Tensor
    unsup_warmup: Tensor

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
        return {
            "util_ratio": self.util_ratio,
            "unsup_warmup": self.unsup_warmup,
        }


def build_pseudolabel_objective_hooks() -> SslObjectiveHooks:
    """PseudoLabel baseline의 tensor-level hook 조합을 만든다."""

    return SslObjectiveHooks(
        pseudo_labeling=HardOrSoftPseudoLabelingHook(),
        masking=FixedThresholdMaskingHook(),
        consistency_loss=CrossEntropyConsistencyLossHook(),
    )


@dataclass(slots=True)
class PseudoLabelAlgorithm:
    """USB PseudoLabel을 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    config: PseudoLabelConfig
    hooks: SslObjectiveHooks = field(default_factory=build_pseudolabel_objective_hooks)
    algorithm_name: str = "pseudolabel"
    _iteration: int = field(default=0, init=False)
    _num_train_iter: int = field(default=1, init=False)

    @property
    def uses_labeled_batches(self) -> bool:
        return self.config.supervised_loss_weight > 0

    def configure_training(self, *, num_train_iter: int) -> None:
        """USB `self.num_train_iter`에 해당하는 warm-up denominator를 설정한다."""

        if num_train_iter <= 0:
            raise ValueError("num_train_iter must be positive.")
        self._num_train_iter = int(num_train_iter)
        self._iteration = 0

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        if unlabeled_loader_length == 0:
            raise ValueError("PseudoLabel unlabeled_loader must not be empty.")
        if self.config.supervised_loss_weight > 0 and train_loader_length == 0:
            raise ValueError(
                "PseudoLabel labeled train_loader must not be empty when "
                "supervised_loss_weight > 0."
            )

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Tensor],
    ) -> QuerySslStepOutput:
        output = compute_pseudolabel_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            config=self.config,
            hooks=self.hooks,
            iteration=self._iteration,
            num_train_iter=self._num_train_iter,
        )
        self._iteration += 1
        return output


def compute_pseudolabel_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Tensor],
    config: PseudoLabelConfig,
    iteration: int,
    num_train_iter: int,
    hooks: SslObjectiveHooks | None = None,
) -> PseudoLabelStepOutput:
    """USB `semilearn/algorithms/pseudolabel/pseudolabel.py::train_step` 핵심."""

    if num_train_iter <= 0:
        raise ValueError("num_train_iter must be positive.")
    if iteration < 0:
        raise ValueError("iteration must not be negative.")

    if labeled_batch is not None:
        logits_x_lb = model(
            input_ids=labeled_batch["input_ids"],
            attention_mask=labeled_batch["attention_mask"],
        )
        sup_loss = F.cross_entropy(
            logits_x_lb, labeled_batch["labels"], reduction="mean"
        )
        del logits_x_lb
    else:
        sup_loss = None

    logits_x_ulb = model(
        input_ids=unlabeled_batch["weak_input_ids"],
        attention_mask=unlabeled_batch["weak_attention_mask"],
    )
    if sup_loss is None:
        sup_loss = logits_x_ulb.new_zeros(())

    effective_hooks = hooks or build_pseudolabel_objective_hooks()
    probs_x_ulb = compute_prob(logits_x_ulb.detach())
    mask = effective_hooks.masking.build_mask(
        probs_x_ulb_w=probs_x_ulb,
        p_cutoff=config.p_cutoff,
    )
    pseudo_label = effective_hooks.pseudo_labeling.generate_targets(
        probs_x_ulb_w=logits_x_ulb,
        config=PseudoLabelingConfig(use_hard_label=True),
    )
    unsup_loss = effective_hooks.consistency_loss.compute_loss(
        logits=logits_x_ulb,
        targets=pseudo_label,
        mask=mask,
    )
    unsup_warmup = logits_x_ulb.new_tensor(
        _compute_usb_unsup_warmup(
            iteration=iteration,
            unsup_warm_up=config.unsup_warm_up,
            num_train_iter=num_train_iter,
        )
    )
    total_loss = (
        config.supervised_loss_weight * sup_loss
        + config.lambda_u * unsup_loss * unsup_warmup
    )
    return PseudoLabelStepOutput(
        total_loss=total_loss,
        sup_loss=sup_loss,
        unsup_loss=unsup_loss,
        mask=mask,
        unsup_warmup=unsup_warmup,
    )


def _compute_usb_unsup_warmup(
    *,
    iteration: int,
    unsup_warm_up: float,
    num_train_iter: int,
) -> float:
    if unsup_warm_up <= 0:
        return 1.0
    denominator = float(unsup_warm_up) * float(num_train_iter)
    return max(0.0, min(1.0, float(iteration) / denominator))


@register_query_ssl_algorithm(
    "pseudolabel",
    "pseudo_label",
    display_name="PseudoLabel",
    required_views=QuerySslRequiredViews(
        view_names=("weak_text",),
        view_builder_name="usb_weak",
    ),
    default_uses_labeled_batches=True,
)
def build_pseudolabel_algorithm(parameters: Mapping[str, Any]) -> PseudoLabelAlgorithm:
    """Hydra method parameter mapping으로 PseudoLabel algorithm을 만든다."""

    return PseudoLabelAlgorithm(
        config=PseudoLabelConfig(
            p_cutoff=float(parameters["p_cutoff"]),
            unsup_warm_up=float(parameters.get("unsup_warm_up", 0.4)),
            lambda_u=float(parameters.get("lambda_u", 1.0)),
            supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
        )
    )

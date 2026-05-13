"""USB PseudoLabel core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor
from torch.nn import functional as F

from ...base import QuerySslRequiredViews, QuerySslStepResult, TextBatchClassifier
from ...common import compute_prob
from ...hooks.consistency import CrossEntropyConsistencyLossHook
from ...hooks.masking import FixedThresholdMaskingHook
from ...hooks.objective import SslObjectiveHooks
from ...hooks.pseudo_labeling import (
    HardOrSoftPseudoLabelingHook,
    PseudoLabelingConfig,
)
from ...registry import register_query_ssl_algorithm
from ...state import (
    build_query_ssl_algorithm_state,
    require_query_ssl_algorithm_state,
)


def build_pseudolabel_objective_hooks() -> SslObjectiveHooks:
    """PseudoLabel baseline의 tensor-level hook 조합을 만든다."""

    return SslObjectiveHooks(
        pseudo_labeling=HardOrSoftPseudoLabelingHook(),
        masking=FixedThresholdMaskingHook(),
        consistency_loss=CrossEntropyConsistencyLossHook(),
    )


class PseudoLabelAlgorithm:
    """USB PseudoLabel을 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "pseudolabel"

    def __init__(
        self,
        *,
        p_cutoff: float,
        unsup_warm_up: float = 0.4,
        lambda_u: float = 1.0,
        supervised_loss_weight: float = 1.0,
        hooks: SslObjectiveHooks | None = None,
    ) -> None:
        self.p_cutoff = float(p_cutoff)
        self.unsup_warm_up = float(unsup_warm_up)
        self.lambda_u = float(lambda_u)
        self.supervised_loss_weight = float(supervised_loss_weight)
        self.hooks = hooks or build_pseudolabel_objective_hooks()
        self._iteration = 0
        self._num_train_iter = 1

    @property
    def uses_labeled_batches(self) -> bool:
        return self.supervised_loss_weight > 0

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
        if self.supervised_loss_weight > 0 and train_loader_length == 0:
            raise ValueError(
                "PseudoLabel labeled train_loader must not be empty when "
                "supervised_loss_weight > 0."
            )

    def export_state(self) -> Mapping[str, Any]:
        """중단 재개용 PseudoLabel warm-up iteration state를 내보낸다."""

        return build_query_ssl_algorithm_state(
            algorithm_name=self.algorithm_name,
            configured=True,
            metadata={
                "iteration": self._iteration,
                "num_train_iter": self._num_train_iter,
            },
        )

    def load_state(self, state: Mapping[str, Any]) -> None:
        """저장된 PseudoLabel warm-up iteration state를 복원한다."""

        state = require_query_ssl_algorithm_state(
            state=state,
            algorithm_name=self.algorithm_name,
        )
        self._iteration = int(state.get("iteration", self._iteration))
        self._num_train_iter = int(state.get("num_train_iter", self._num_train_iter))

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Tensor],
    ) -> QuerySslStepResult:
        output = compute_pseudolabel_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            p_cutoff=self.p_cutoff,
            unsup_warm_up=self.unsup_warm_up,
            lambda_u=self.lambda_u,
            supervised_loss_weight=self.supervised_loss_weight,
            iteration=self._iteration,
            num_train_iter=self._num_train_iter,
            hooks=self.hooks,
        )
        self._iteration += 1
        return output


def compute_pseudolabel_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Tensor],
    p_cutoff: float,
    iteration: int,
    num_train_iter: int,
    unsup_warm_up: float = 0.4,
    lambda_u: float = 1.0,
    supervised_loss_weight: float = 1.0,
    hooks: SslObjectiveHooks | None = None,
) -> QuerySslStepResult:
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
        p_cutoff=p_cutoff,
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
            unsup_warm_up=unsup_warm_up,
            num_train_iter=num_train_iter,
        )
    )
    total_loss = (
        supervised_loss_weight * sup_loss + lambda_u * unsup_loss * unsup_warmup
    )
    return QuerySslStepResult(
        total_loss=total_loss,
        loss_components={
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
        },
        metrics={
            "util_ratio": mask.float().mean(),
            "unsup_warmup": unsup_warmup,
        },
        debug_tensors={"mask": mask},
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
        p_cutoff=float(parameters["p_cutoff"]),
        unsup_warm_up=float(parameters.get("unsup_warm_up", 0.4)),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
        supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
    )

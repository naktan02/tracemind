"""USB SoftMatch core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor
from torch.nn import functional as F

from ...base import (
    QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA,
    QUERY_SSL_ALGORITHM_STATE_WEIGHTING_EMA,
    QuerySslRuntimeRequirements,
    QuerySslStepResult,
    TextBatchClassifier,
)
from ...hooks.consistency import ConsistencyLossHook, CrossEntropyConsistencyLossHook
from ...hooks.distribution_alignment import EmaDistributionAlignmentHook
from ...hooks.pseudo_labeling import (
    HardOrSoftPseudoLabelingHook,
    PseudoLabelingConfig,
    PseudoLabelingHook,
)
from ...primitives.probability import compute_prob
from ...registry import register_query_ssl_algorithm
from ...state import (
    build_query_ssl_algorithm_state,
    is_configured_query_ssl_algorithm_state,
    load_tensor_state_field,
    require_matching_int_state_value,
    require_query_ssl_algorithm_state,
)
from ..usb_consistency import (
    USB_MULTIVIEW_REQUIRED_VIEWS,
    compute_unlabeled_weak_strong_logits,
)
from .weighting import SoftMatchWeightingHook


class SoftMatchAlgorithm:
    """SoftMatch를 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "softmatch"

    def __init__(
        self,
        *,
        temperature: float,
        hard_label: bool = True,
        dist_align: bool = True,
        dist_uniform: bool = True,
        ema_p: float = 0.999,
        n_sigma: float = 2.0,
        per_class: bool = False,
        lambda_u: float = 1.0,
        supervised_loss_weight: float = 1.0,
        pseudo_labeling_hook: PseudoLabelingHook | None = None,
        consistency_loss_hook: ConsistencyLossHook | None = None,
    ) -> None:
        self.temperature = float(temperature)
        self.hard_label = bool(hard_label)
        self.dist_align = bool(dist_align)
        self.dist_uniform = bool(dist_uniform)
        self.ema_p = float(ema_p)
        self.n_sigma = float(n_sigma)
        self.per_class = bool(per_class)
        self.lambda_u = float(lambda_u)
        self.supervised_loss_weight = float(supervised_loss_weight)
        self.pseudo_labeling_hook = (
            pseudo_labeling_hook or HardOrSoftPseudoLabelingHook()
        )
        self.consistency_loss_hook = (
            consistency_loss_hook or CrossEntropyConsistencyLossHook()
        )
        self.num_classes: int | None = None
        self.dist_align_hook: EmaDistributionAlignmentHook | None = None
        self.weighting_hook: SoftMatchWeightingHook | None = None

    @property
    def uses_labeled_batches(self) -> bool:
        return self.supervised_loss_weight > 0 or (
            self.dist_align and not self.dist_uniform
        )

    def configure_dataset(
        self,
        *,
        num_classes: int,
        unlabeled_row_count: int,
    ) -> None:
        """USB `num_classes` 기반 EMA DA/weighting state를 초기화한다."""

        del unlabeled_row_count
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = int(num_classes)
        self.dist_align_hook = (
            EmaDistributionAlignmentHook(
                num_classes=self.num_classes,
                momentum=self.ema_p,
                p_target_type="uniform" if self.dist_uniform else "model",
            )
            if self.dist_align
            else None
        )
        self.weighting_hook = SoftMatchWeightingHook(
            num_classes=self.num_classes,
            n_sigma=self.n_sigma,
            momentum=self.ema_p,
            per_class=self.per_class,
        )

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        if unlabeled_loader_length == 0:
            raise ValueError("SoftMatch unlabeled_loader must not be empty.")
        if self.uses_labeled_batches and train_loader_length == 0:
            raise ValueError(
                "SoftMatch labeled train_loader must not be empty when supervised "
                "loss or model-target distribution alignment is enabled."
            )

    def export_state(self) -> Mapping[str, Any]:
        """중단 재개용 SoftMatch EMA state를 내보낸다."""

        if self.weighting_hook is None:
            return build_query_ssl_algorithm_state(
                algorithm_name=self.algorithm_name,
                configured=False,
            )
        tensors: dict[str, Tensor | None] = {
            "prob_max_mu_t": self.weighting_hook.prob_max_mu_t,
            "prob_max_var_t": self.weighting_hook.prob_max_var_t,
        }
        metadata: dict[str, Any] = {
            "num_classes": self.num_classes,
            "ema_p": self.ema_p,
            "n_sigma": self.n_sigma,
            "per_class": self.per_class,
            "dist_align": self.dist_align,
            "dist_uniform": self.dist_uniform,
        }
        if self.dist_align_hook is not None:
            metadata["p_target_type"] = self.dist_align_hook.p_target_type
            tensors["p_model"] = self.dist_align_hook.p_model
            tensors["p_target"] = self.dist_align_hook.p_target
        return build_query_ssl_algorithm_state(
            algorithm_name=self.algorithm_name,
            configured=True,
            metadata=metadata,
            tensors=tensors,
        )

    def load_state(self, state: Mapping[str, Any]) -> None:
        """저장된 SoftMatch EMA state를 복원한다."""

        if self.weighting_hook is None:
            raise ValueError("SoftMatch requires configure_dataset before load_state.")
        state = require_query_ssl_algorithm_state(
            state=state,
            algorithm_name=self.algorithm_name,
        )
        if not is_configured_query_ssl_algorithm_state(state):
            return
        if self.num_classes is None:
            raise ValueError("SoftMatch dataset metadata is not configured.")
        require_matching_int_state_value(
            state=state,
            field_name="num_classes",
            expected=self.num_classes,
            algorithm_name="SoftMatch",
        )
        device = self.weighting_hook.prob_max_mu_t.device
        prob_max_mu_t = load_tensor_state_field(
            state=state,
            field_name="prob_max_mu_t",
            device=device,
            algorithm_name="SoftMatch",
        )
        prob_max_var_t = load_tensor_state_field(
            state=state,
            field_name="prob_max_var_t",
            device=device,
            algorithm_name="SoftMatch",
        )
        assert prob_max_mu_t is not None
        assert prob_max_var_t is not None
        self.weighting_hook.prob_max_mu_t = prob_max_mu_t
        self.weighting_hook.prob_max_var_t = prob_max_var_t
        if self.dist_align_hook is not None:
            self.dist_align_hook.p_model = load_tensor_state_field(
                state=state,
                field_name="p_model",
                device=device,
                algorithm_name="SoftMatch",
                allow_none=True,
            )
            p_target = load_tensor_state_field(
                state=state,
                field_name="p_target",
                device=device,
                algorithm_name="SoftMatch",
            )
            assert p_target is not None
            self.dist_align_hook.p_target = p_target

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepResult:
        if self.weighting_hook is None:
            raise ValueError(
                "SoftMatch requires configure_dataset before compute_step."
            )
        return compute_softmatch_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            temperature=self.temperature,
            hard_label=self.hard_label,
            lambda_u=self.lambda_u,
            supervised_loss_weight=self.supervised_loss_weight,
            pseudo_labeling_hook=self.pseudo_labeling_hook,
            consistency_loss_hook=self.consistency_loss_hook,
            dist_align_hook=self.dist_align_hook,
            weighting_hook=self.weighting_hook,
        )


def compute_softmatch_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Any],
    temperature: float,
    hard_label: bool = True,
    lambda_u: float = 1.0,
    supervised_loss_weight: float = 1.0,
    pseudo_labeling_hook: PseudoLabelingHook | None = None,
    consistency_loss_hook: ConsistencyLossHook | None = None,
    dist_align_hook: EmaDistributionAlignmentHook | None,
    weighting_hook: SoftMatchWeightingHook,
) -> QuerySslStepResult:
    """USB `semilearn/algorithms/softmatch/softmatch.py::train_step` 핵심."""

    logits_x_lb: Tensor | None = None
    if labeled_batch is not None:
        logits_x_lb = model(
            input_ids=labeled_batch["input_ids"],
            attention_mask=labeled_batch["attention_mask"],
        )
    logits_x_ulb_s, logits_x_ulb_w = compute_unlabeled_weak_strong_logits(
        model=model,
        unlabeled_batch=unlabeled_batch,
    )
    if logits_x_lb is None:
        if supervised_loss_weight > 0:
            raise ValueError("SoftMatch requires labeled_batch for supervised loss.")
        sup_loss = logits_x_ulb_s.new_zeros(())
    else:
        sup_loss = F.cross_entropy(
            logits_x_lb, labeled_batch["labels"], reduction="mean"
        )

    probs_x_lb = None if logits_x_lb is None else compute_prob(logits_x_lb.detach())
    probs_x_ulb_w = compute_prob(logits_x_ulb_w.detach())
    aligned_probs_x_ulb_w = probs_x_ulb_w
    if dist_align_hook is not None:
        aligned_probs_x_ulb_w = dist_align_hook.dist_align(
            probs_x_ulb=probs_x_ulb_w,
            probs_x_lb=probs_x_lb,
        )

    mask = weighting_hook.masking(
        logits_x_ulb=aligned_probs_x_ulb_w,
        softmax_x_ulb=False,
    )
    effective_pseudo_labeling_hook = (
        pseudo_labeling_hook or HardOrSoftPseudoLabelingHook()
    )
    pseudo_label = effective_pseudo_labeling_hook.generate_targets_from_logits(
        logits_x_ulb_w=logits_x_ulb_w,
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
    debug_tensors = {
        "mask": mask,
        "prob_max_mu_t": weighting_hook.prob_max_mu_t,
        "prob_max_var_t": weighting_hook.prob_max_var_t,
        "probs_x_ulb_w_aligned": aligned_probs_x_ulb_w,
    }
    if dist_align_hook is not None and dist_align_hook.p_model is not None:
        debug_tensors["p_model"] = dist_align_hook.p_model
        debug_tensors["p_target"] = dist_align_hook.p_target
    return QuerySslStepResult(
        total_loss=total_loss,
        loss_components={
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
        },
        metrics={"util_ratio": mask.float().mean()},
        debug_tensors=debug_tensors,
    )


@register_query_ssl_algorithm(
    "softmatch",
    display_name="SoftMatch",
    required_views=USB_MULTIVIEW_REQUIRED_VIEWS,
    default_uses_labeled_batches=True,
    runtime_requirements=QuerySslRuntimeRequirements(
        algorithm_state_surface=frozenset(
            {
                QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA,
                QUERY_SSL_ALGORITHM_STATE_WEIGHTING_EMA,
            }
        ),
    ),
)
def build_softmatch_algorithm(parameters: Mapping[str, Any]) -> SoftMatchAlgorithm:
    """Hydra method parameter mapping으로 SoftMatch algorithm을 만든다."""

    return SoftMatchAlgorithm(
        temperature=float(parameters["temperature"]),
        hard_label=bool(parameters.get("hard_label", True)),
        dist_align=bool(parameters.get("dist_align", True)),
        dist_uniform=bool(parameters.get("dist_uniform", True)),
        ema_p=float(parameters.get("ema_p", 0.999)),
        n_sigma=float(parameters.get("n_sigma", 2.0)),
        per_class=bool(parameters.get("per_class", False)),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
        supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
    )

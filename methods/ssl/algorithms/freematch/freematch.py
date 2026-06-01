"""USB FreeMatch core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from ...base import (
    QUERY_SSL_ALGORITHM_STATE_ADAPTIVE_THRESHOLD,
    QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA,
    QuerySslRuntimeRequirements,
    QuerySslStepResult,
    TextBatchClassifier,
)
from ...hooks.adaptive_thresholding import (
    FreeMatchThresholdingHook,
    replace_inf_to_zero,
)
from ...hooks.consistency import ConsistencyLossHook, CrossEntropyConsistencyLossHook
from ...hooks.pseudo_labeling import (
    HardOrSoftPseudoLabelingHook,
    PseudoLabelingConfig,
    PseudoLabelingHook,
)
from ...hooks.supervised import compute_labeled_cross_entropy_loss
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
    validate_usb_consistency_loaders,
)


def entropy_loss(
    mask: Tensor,
    logits_s: Tensor,
    prob_model: Tensor,
    label_hist: Tensor,
) -> tuple[Tensor, Tensor]:
    """USB FreeMatch `entropy_loss`의 self-adaptive fairness regularization."""

    mask = mask.bool()
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(
        logits_s.dtype
    )
    hist_s = hist_s / hist_s.sum()

    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


class FreeMatchAlgorithm:
    """FreeMatch를 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "freematch"

    def __init__(
        self,
        *,
        temperature: float,
        hard_label: bool = True,
        ema_p: float = 0.999,
        ent_loss_ratio: float = 0.01,
        use_quantile: bool = False,
        clip_thresh: bool = False,
        lambda_u: float = 1.0,
        supervised_loss_weight: float = 1.0,
        pseudo_labeling_hook: PseudoLabelingHook | None = None,
        consistency_loss_hook: ConsistencyLossHook | None = None,
    ) -> None:
        self.temperature = float(temperature)
        self.hard_label = bool(hard_label)
        self.ema_p = float(ema_p)
        self.ent_loss_ratio = float(ent_loss_ratio)
        self.use_quantile = bool(use_quantile)
        self.clip_thresh = bool(clip_thresh)
        self.lambda_u = float(lambda_u)
        self.supervised_loss_weight = float(supervised_loss_weight)
        self.pseudo_labeling_hook = (
            pseudo_labeling_hook or HardOrSoftPseudoLabelingHook()
        )
        self.consistency_loss_hook = (
            consistency_loss_hook or CrossEntropyConsistencyLossHook()
        )
        self.masking_hook: FreeMatchThresholdingHook | None = None
        self.p_model: Tensor | None = None
        self.label_hist: Tensor | None = None
        self.time_p: Tensor | None = None

    @property
    def uses_labeled_batches(self) -> bool:
        return self.supervised_loss_weight > 0

    def configure_dataset(
        self,
        *,
        num_classes: int,
        unlabeled_row_count: int,
    ) -> None:
        """USB `num_classes` 기반 masking hook state를 초기화한다."""

        del unlabeled_row_count
        self.masking_hook = FreeMatchThresholdingHook(
            num_classes=num_classes,
            momentum=self.ema_p,
        )
        self.p_model = self.masking_hook.p_model
        self.label_hist = self.masking_hook.label_hist
        self.time_p = self.masking_hook.time_p

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        validate_usb_consistency_loaders(
            algorithm_name="FreeMatch",
            train_loader_length=train_loader_length,
            unlabeled_loader_length=unlabeled_loader_length,
            supervised_loss_weight=self.supervised_loss_weight,
        )

    def export_state(self) -> Mapping[str, Any]:
        """중단 재개용 FreeMatch thresholding state를 내보낸다."""

        if self.masking_hook is None:
            return build_query_ssl_algorithm_state(
                algorithm_name=self.algorithm_name,
                configured=False,
            )
        return build_query_ssl_algorithm_state(
            algorithm_name=self.algorithm_name,
            configured=True,
            metadata={
                "num_classes": self.masking_hook.num_classes,
                "ema_p": self.ema_p,
            },
            tensors={
                "p_model": self.masking_hook.p_model,
                "label_hist": self.masking_hook.label_hist,
                "time_p": self.masking_hook.time_p,
            },
        )

    def load_state(self, state: Mapping[str, Any]) -> None:
        """저장된 FreeMatch thresholding state를 복원한다."""

        if self.masking_hook is None:
            raise ValueError("FreeMatch requires configure_dataset before load_state.")
        state = require_query_ssl_algorithm_state(
            state=state,
            algorithm_name=self.algorithm_name,
        )
        if not is_configured_query_ssl_algorithm_state(state):
            return
        require_matching_int_state_value(
            state=state,
            field_name="num_classes",
            expected=self.masking_hook.num_classes,
            algorithm_name="FreeMatch",
        )
        device = self.masking_hook.p_model.device
        p_model = load_tensor_state_field(
            state=state,
            field_name="p_model",
            device=device,
            algorithm_name="FreeMatch",
        )
        label_hist = load_tensor_state_field(
            state=state,
            field_name="label_hist",
            device=device,
            algorithm_name="FreeMatch",
        )
        time_p = load_tensor_state_field(
            state=state,
            field_name="time_p",
            device=device,
            algorithm_name="FreeMatch",
        )
        assert p_model is not None
        assert label_hist is not None
        assert time_p is not None
        self.masking_hook.p_model = p_model
        self.masking_hook.label_hist = label_hist
        self.masking_hook.time_p = time_p
        self.p_model = self.masking_hook.p_model
        self.label_hist = self.masking_hook.label_hist
        self.time_p = self.masking_hook.time_p

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepResult:
        if self.masking_hook is None:
            raise ValueError(
                "FreeMatch requires configure_dataset before compute_step."
            )
        return compute_freematch_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            temperature=self.temperature,
            hard_label=self.hard_label,
            lambda_u=self.lambda_u,
            ent_loss_ratio=self.ent_loss_ratio,
            supervised_loss_weight=self.supervised_loss_weight,
            pseudo_labeling_hook=self.pseudo_labeling_hook,
            consistency_loss_hook=self.consistency_loss_hook,
            masking_hook=self.masking_hook,
            algorithm=self,
        )


def compute_freematch_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Any],
    temperature: float,
    hard_label: bool = True,
    lambda_u: float = 1.0,
    ent_loss_ratio: float = 0.01,
    supervised_loss_weight: float = 1.0,
    pseudo_labeling_hook: PseudoLabelingHook | None = None,
    consistency_loss_hook: ConsistencyLossHook | None = None,
    masking_hook: FreeMatchThresholdingHook,
    algorithm: FreeMatchAlgorithm | None = None,
) -> QuerySslStepResult:
    """USB `semilearn/algorithms/freematch/freematch.py::train_step` 핵심."""

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

    masking_algorithm = algorithm or _FreeMatchMaskingAlgorithm()
    mask = masking_hook.masking(
        masking_algorithm,
        logits_x_ulb=logits_x_ulb_w,
        softmax_x_ulb=True,
    )
    pseudo_label = build_freematch_pseudo_label(
        pseudo_labeling=pseudo_labeling_hook or HardOrSoftPseudoLabelingHook(),
        logits_x_ulb_w=logits_x_ulb_w,
        hard_label=hard_label,
        temperature=temperature,
    )
    effective_consistency_loss_hook = (
        consistency_loss_hook or CrossEntropyConsistencyLossHook()
    )
    unsup_loss = effective_consistency_loss_hook.compute_loss(
        logits=logits_x_ulb_s,
        targets=pseudo_label,
        mask=mask,
    )
    if mask.sum() > 0:
        ent_loss, _ = entropy_loss(
            mask,
            logits_x_ulb_s,
            masking_hook.p_model,
            masking_hook.label_hist,
        )
    else:
        ent_loss = logits_x_ulb_s.new_zeros(())
    total_loss = (
        supervised_loss_weight * sup_loss
        + lambda_u * unsup_loss
        + ent_loss_ratio * ent_loss
    )
    return QuerySslStepResult(
        total_loss=total_loss,
        loss_components={
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "ent_loss": ent_loss,
        },
        metrics={
            "util_ratio": mask.float().mean(),
            "time_p": masking_hook.time_p,
        },
        debug_tensors={
            "mask": mask,
            "p_model": masking_hook.p_model,
            "label_hist": masking_hook.label_hist,
        },
    )


class _FreeMatchMaskingAlgorithm:
    use_quantile = False
    clip_thresh = False


def build_freematch_pseudo_label(
    *,
    pseudo_labeling: PseudoLabelingHook | None = None,
    logits_x_ulb_w: Tensor,
    hard_label: bool,
    temperature: float,
) -> Tensor:
    """USB FreeMatch `PseudoLabelingHook(..., softmax=True)` 경로."""

    effective_pseudo_labeling = pseudo_labeling or HardOrSoftPseudoLabelingHook()
    return effective_pseudo_labeling.generate_targets_from_logits(
        logits_x_ulb_w=logits_x_ulb_w,
        config=PseudoLabelingConfig(
            use_hard_label=hard_label,
            temperature=temperature,
        ),
    )


@register_query_ssl_algorithm(
    "freematch",
    display_name="FreeMatch",
    required_views=USB_MULTIVIEW_REQUIRED_VIEWS,
    default_uses_labeled_batches=True,
    runtime_requirements=QuerySslRuntimeRequirements(
        algorithm_state_surface=frozenset(
            {
                QUERY_SSL_ALGORITHM_STATE_ADAPTIVE_THRESHOLD,
                QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA,
            }
        ),
    ),
)
def build_freematch_algorithm(parameters: Mapping[str, Any]) -> FreeMatchAlgorithm:
    """Hydra method parameter mapping으로 FreeMatch algorithm을 만든다."""

    return FreeMatchAlgorithm(
        temperature=float(parameters["temperature"]),
        hard_label=bool(parameters.get("hard_label", True)),
        ema_p=float(parameters.get("ema_p", 0.999)),
        ent_loss_ratio=float(parameters.get("ent_loss_ratio", 0.01)),
        use_quantile=bool(parameters.get("use_quantile", False)),
        clip_thresh=bool(parameters.get("clip_thresh", False)),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
        supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
    )

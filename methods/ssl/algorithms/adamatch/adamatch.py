"""USB AdaMatch core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor
from torch.nn import functional as F

from ...base import (
    QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA,
    QuerySslRuntimeRequirements,
    QuerySslStepResult,
    TextBatchClassifier,
)
from ...hooks.adaptive_thresholding import RelativeConfidenceThresholdingHook
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

AdaMatchThresholdingHook = RelativeConfidenceThresholdingHook


class AdaMatchDistAlignHook(EmaDistributionAlignmentHook):
    """USB AdaMatch가 쓰는 EMA distribution alignment 조합."""

    hook_name: str = "adamatch_dist_align_ema"

    def __init__(
        self,
        *,
        num_classes: int,
        momentum: float = 0.999,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            momentum=momentum,
            p_target_type="model",
        )


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

    def export_state(self) -> Mapping[str, Any]:
        """중단 재개용 AdaMatch distribution-alignment state를 내보낸다."""

        if self.dist_align_hook is None:
            return build_query_ssl_algorithm_state(
                algorithm_name=self.algorithm_name,
                configured=False,
            )
        return build_query_ssl_algorithm_state(
            algorithm_name=self.algorithm_name,
            configured=True,
            metadata={
                "num_classes": self.dist_align_hook.num_classes,
                "ema_p": self.ema_p,
            },
            tensors={
                "p_model": self.dist_align_hook.p_model,
                "p_target": self.dist_align_hook.p_target,
            },
        )

    def load_state(self, state: Mapping[str, Any]) -> None:
        """저장된 AdaMatch distribution-alignment state를 복원한다."""

        if self.dist_align_hook is None:
            raise ValueError("AdaMatch requires configure_dataset before load_state.")
        state = require_query_ssl_algorithm_state(
            state=state,
            algorithm_name=self.algorithm_name,
        )
        if not is_configured_query_ssl_algorithm_state(state):
            return
        require_matching_int_state_value(
            state=state,
            field_name="num_classes",
            expected=self.dist_align_hook.num_classes,
            algorithm_name="AdaMatch",
        )
        device = self.dist_align_hook.p_target.device
        self.dist_align_hook.p_model = load_tensor_state_field(
            state=state,
            field_name="p_model",
            device=device,
            algorithm_name="AdaMatch",
            allow_none=True,
        )
        p_target = load_tensor_state_field(
            state=state,
            field_name="p_target",
            device=device,
            algorithm_name="AdaMatch",
        )
        assert p_target is not None
        self.dist_align_hook.p_target = p_target
        self.p_model = self.dist_align_hook.p_model
        self.p_target = self.dist_align_hook.p_target

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepResult:
        if self.dist_align_hook is None:
            raise ValueError("AdaMatch requires configure_dataset before compute_step.")
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
    runtime_requirements=QuerySslRuntimeRequirements(
        algorithm_state_surface=frozenset({QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA}),
    ),
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

"""USB FlexMatch core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor

from ...base import (
    QUERY_SSL_ALGORITHM_STATE_ADAPTIVE_THRESHOLD,
    QUERY_SSL_ALGORITHM_STATE_DATASET_STATE,
    QuerySslRuntimeRequirements,
    QuerySslStepResult,
    TextBatchClassifier,
)
from ...hooks.consistency import ConsistencyLossHook, CrossEntropyConsistencyLossHook
from ...hooks.pseudo_labeling import (
    HardOrSoftPseudoLabelingHook,
    PseudoLabelingConfig,
    PseudoLabelingHook,
)
from ...hooks.supervised import compute_labeled_cross_entropy_loss
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
    validate_usb_consistency_loaders,
)
from .thresholding import FlexMatchThresholdingHook


class FlexMatchAlgorithm:
    """FlexMatch를 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "flexmatch"

    def __init__(
        self,
        *,
        temperature: float,
        p_cutoff: float,
        hard_label: bool = True,
        thresh_warmup: bool = True,
        lambda_u: float = 1.0,
        supervised_loss_weight: float = 1.0,
        pseudo_labeling_hook: PseudoLabelingHook | None = None,
        consistency_loss_hook: ConsistencyLossHook | None = None,
    ) -> None:
        self.temperature = float(temperature)
        self.p_cutoff = float(p_cutoff)
        self.hard_label = bool(hard_label)
        self.thresh_warmup = bool(thresh_warmup)
        self.lambda_u = float(lambda_u)
        self.supervised_loss_weight = float(supervised_loss_weight)
        self.pseudo_labeling_hook = (
            pseudo_labeling_hook or HardOrSoftPseudoLabelingHook()
        )
        self.consistency_loss_hook = (
            consistency_loss_hook or CrossEntropyConsistencyLossHook()
        )
        self.masking_hook: FlexMatchThresholdingHook | None = None

    @property
    def uses_labeled_batches(self) -> bool:
        return self.supervised_loss_weight > 0

    def configure_dataset(
        self,
        *,
        num_classes: int,
        unlabeled_row_count: int,
    ) -> None:
        """USB `ulb_dest_len`/`num_classes` 기반 hook state를 초기화한다."""

        self.masking_hook = FlexMatchThresholdingHook(
            ulb_dest_len=unlabeled_row_count,
            num_classes=num_classes,
            thresh_warmup=self.thresh_warmup,
        )

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        validate_usb_consistency_loaders(
            algorithm_name="FlexMatch",
            train_loader_length=train_loader_length,
            unlabeled_loader_length=unlabeled_loader_length,
            supervised_loss_weight=self.supervised_loss_weight,
        )

    def export_state(self) -> Mapping[str, Any]:
        """중단 재개용 FlexMatch classwise threshold state를 내보낸다."""

        if self.masking_hook is None:
            return build_query_ssl_algorithm_state(
                algorithm_name=self.algorithm_name,
                configured=False,
            )
        return build_query_ssl_algorithm_state(
            algorithm_name=self.algorithm_name,
            configured=True,
            metadata={
                "ulb_dest_len": self.masking_hook.ulb_dest_len,
                "num_classes": self.masking_hook.num_classes,
                "thresh_warmup": self.masking_hook.thresh_warmup,
            },
            tensors={
                "selected_label": self.masking_hook.selected_label,
                "classwise_acc": self.masking_hook.classwise_acc,
            },
        )

    def load_state(self, state: Mapping[str, Any]) -> None:
        """저장된 FlexMatch classwise threshold state를 복원한다."""

        if self.masking_hook is None:
            raise ValueError("FlexMatch requires configure_dataset before load_state.")
        state = require_query_ssl_algorithm_state(
            state=state,
            algorithm_name=self.algorithm_name,
        )
        if not is_configured_query_ssl_algorithm_state(state):
            return
        require_matching_int_state_value(
            state=state,
            field_name="ulb_dest_len",
            expected=self.masking_hook.ulb_dest_len,
            algorithm_name="FlexMatch",
        )
        require_matching_int_state_value(
            state=state,
            field_name="num_classes",
            expected=self.masking_hook.num_classes,
            algorithm_name="FlexMatch",
        )
        device = self.masking_hook.classwise_acc.device
        selected_label = load_tensor_state_field(
            state=state,
            field_name="selected_label",
            device=device,
            algorithm_name="FlexMatch",
        )
        classwise_acc = load_tensor_state_field(
            state=state,
            field_name="classwise_acc",
            device=device,
            algorithm_name="FlexMatch",
        )
        assert selected_label is not None
        assert classwise_acc is not None
        self.masking_hook.selected_label = selected_label
        self.masking_hook.classwise_acc = classwise_acc

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepResult:
        if self.masking_hook is None:
            raise ValueError(
                "FlexMatch requires configure_dataset before compute_step."
            )
        return compute_flexmatch_step(
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
            algorithm=self,
        )


def compute_flexmatch_step(
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
    masking_hook: FlexMatchThresholdingHook,
    algorithm: FlexMatchAlgorithm | None = None,
) -> QuerySslStepResult:
    """USB `semilearn/algorithms/flexmatch/flexmatch.py::train_step` 핵심."""

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
    max_probs, _ = probs_x_ulb_w.max(dim=-1)
    high_conf_mask = max_probs.ge(p_cutoff)
    masking_algorithm = algorithm or _FlexMatchMaskingAlgorithm(p_cutoff=p_cutoff)
    mask = masking_hook.masking(
        masking_algorithm,
        logits_x_ulb=probs_x_ulb_w,
        softmax_x_ulb=False,
        idx_ulb=_require_row_indices(unlabeled_batch).to(probs_x_ulb_w.device),
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
    return QuerySslStepResult(
        total_loss=total_loss,
        loss_components={
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
        },
        metrics={
            "util_ratio": mask.float().mean(),
            "high_conf_ratio": high_conf_mask.float().mean(),
            "selected_label_coverage": masking_hook.selected_label.ge(0).float().mean(),
            "classwise_acc_mean": masking_hook.classwise_acc.float().mean(),
            "classwise_acc_max": masking_hook.classwise_acc.float().max(),
        },
        debug_tensors={
            "mask": mask,
            "classwise_acc": masking_hook.classwise_acc,
        },
    )


class _FlexMatchMaskingAlgorithm:
    def __init__(self, *, p_cutoff: float) -> None:
        self.p_cutoff = p_cutoff


def _require_row_indices(unlabeled_batch: Mapping[str, Any]) -> Tensor:
    row_indices = unlabeled_batch.get("row_indices")
    if not isinstance(row_indices, Tensor):
        raise ValueError("FlexMatch requires unlabeled_batch['row_indices'].")
    return row_indices.long()


@register_query_ssl_algorithm(
    "flexmatch",
    display_name="FlexMatch",
    required_views=USB_MULTIVIEW_REQUIRED_VIEWS,
    default_uses_labeled_batches=True,
    runtime_requirements=QuerySslRuntimeRequirements(
        algorithm_state_surface=frozenset(
            {
                QUERY_SSL_ALGORITHM_STATE_ADAPTIVE_THRESHOLD,
                QUERY_SSL_ALGORITHM_STATE_DATASET_STATE,
            }
        ),
    ),
)
def build_flexmatch_algorithm(parameters: Mapping[str, Any]) -> FlexMatchAlgorithm:
    """Hydra method parameter mapping으로 FlexMatch algorithm을 만든다."""

    return FlexMatchAlgorithm(
        temperature=float(parameters["temperature"]),
        p_cutoff=float(parameters["p_cutoff"]),
        hard_label=bool(parameters.get("hard_label", True)),
        thresh_warmup=bool(parameters.get("thresh_warmup", True)),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
        supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
    )

"""USB ReMixMatch core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from methods.adaptation.query_text_views.view_rows import (
    USB_WEAK_STRONG_PAIR_BUILDER_NAME,
)

from ...base import (
    QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA,
    QUERY_SSL_BATCH_SURFACE_WEAK_STRONG_PAIR,
    QUERY_SSL_MODEL_OUTPUT_LOGITS,
    QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP,
    QuerySslRequiredViews,
    QuerySslRuntimeRequirements,
    QuerySslStepContext,
    QuerySslStepResult,
    TextBatchClassifier,
)
from ...hooks.distribution_alignment import EmaDistributionAlignmentHook
from ...model_capabilities import (
    classify_classifier_input_features,
    extract_classifier_input_features,
)
from ...primitives.losses import probability_mse_loss, soft_cross_entropy_loss
from ...primitives.mixup import mixup_one_target
from ...primitives.probability import compute_prob, sharpen_probabilities
from ...registry import register_query_ssl_algorithm
from ...runtime.schedules import compute_linear_warmup
from ...state import (
    build_query_ssl_algorithm_state,
    is_configured_query_ssl_algorithm_state,
    load_tensor_state_field,
    require_matching_int_state_value,
    require_query_ssl_algorithm_state,
)
from ..usb_consistency import validate_usb_consistency_loaders

REMIXMATCH_REQUIRED_VIEWS = QuerySslRequiredViews(
    view_names=("text", "aug_0", "aug_1"),
    view_builder_name=USB_WEAK_STRONG_PAIR_BUILDER_NAME,
)


class ReMixMatchAlgorithm:
    """ReMixMatch를 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "remixmatch"

    def __init__(
        self,
        *,
        T: float = 0.5,
        unsup_warm_up: float = 1.0 / 64.0,
        mixup_alpha: float = 0.75,
        mixup_manifold: bool = True,
        kl_loss_ratio: float = 0.5,
        rot_loss_ratio: float = 0.0,
        lambda_u: float = 1.0,
        supervised_loss_weight: float = 1.0,
        dist_align_momentum: float = 0.999,
    ) -> None:
        self.T = _require_positive_float(T, "T")
        self.unsup_warm_up = float(unsup_warm_up)
        self.mixup_alpha = _require_non_negative_float(mixup_alpha, "mixup_alpha")
        self.mixup_manifold = bool(mixup_manifold)
        if not self.mixup_manifold:
            raise ValueError(
                "TraceMind text ReMixMatch supports only mixup_manifold=True."
            )
        self.kl_loss_ratio = float(kl_loss_ratio)
        self.rot_loss_ratio = float(rot_loss_ratio)
        if self.rot_loss_ratio != 0.0:
            raise ValueError("TraceMind text ReMixMatch requires rot_loss_ratio=0.0.")
        self.lambda_u = float(lambda_u)
        self.supervised_loss_weight = float(supervised_loss_weight)
        self.dist_align_momentum = float(dist_align_momentum)
        self.num_classes: int | None = None
        self.class_distribution: Tensor | None = None
        self.dist_align_hook: EmaDistributionAlignmentHook | None = None
        self._num_train_iter = 1
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
        del unlabeled_row_count
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = int(num_classes)
        self._maybe_build_dist_align_hook()

    def configure_labeled_distribution(self, *, class_distribution: Tensor) -> None:
        distribution = class_distribution.detach().to(dtype=torch.float32).reshape(-1)
        if distribution.numel() <= 0:
            raise ValueError("class_distribution must not be empty.")
        if torch.any(distribution < 0):
            raise ValueError("class_distribution must not contain negative values.")
        total = distribution.sum()
        if float(total.item()) <= 0:
            raise ValueError("class_distribution must have positive mass.")
        self.class_distribution = distribution / total
        self._maybe_build_dist_align_hook()

    def configure_training(self, *, num_train_iter: int) -> None:
        """USB `self.num_train_iter`에 해당하는 warm-up denominator를 설정한다."""

        if num_train_iter <= 0:
            raise ValueError("num_train_iter must be positive.")
        self._num_train_iter = int(num_train_iter)

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        validate_usb_consistency_loaders(
            algorithm_name="ReMixMatch",
            train_loader_length=train_loader_length,
            unlabeled_loader_length=unlabeled_loader_length,
            supervised_loss_weight=1.0,
        )

    def export_state(self) -> Mapping[str, Any]:
        """중단 재개용 ReMixMatch EMA DA state를 내보낸다."""

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
                "dist_align_momentum": self.dist_align_momentum,
            },
            tensors={
                "p_model": self.dist_align_hook.p_model,
                "p_target": self.dist_align_hook.p_target,
            },
        )

    def load_state(self, state: Mapping[str, Any]) -> None:
        """저장된 ReMixMatch EMA DA state를 복원한다."""

        if self.dist_align_hook is None:
            raise ValueError("ReMixMatch requires dataset configuration before state.")
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
            algorithm_name="ReMixMatch",
        )
        device = self.dist_align_hook.p_target.device
        self.dist_align_hook.p_model = load_tensor_state_field(
            state=state,
            field_name="p_model",
            device=device,
            algorithm_name="ReMixMatch",
            allow_none=True,
        )
        p_target = load_tensor_state_field(
            state=state,
            field_name="p_target",
            device=device,
            algorithm_name="ReMixMatch",
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
        return self._compute_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            iteration=0,
            num_train_iter=self._num_train_iter,
        )

    def compute_step_with_context(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
        step_context: QuerySslStepContext,
    ) -> QuerySslStepResult:
        return self._compute_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            iteration=step_context.global_step - 1,
            num_train_iter=step_context.total_train_steps,
        )

    def _compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
        iteration: int,
        num_train_iter: int,
    ) -> QuerySslStepResult:
        if self.num_classes is None or self.dist_align_hook is None:
            raise ValueError(
                "ReMixMatch requires dataset and labeled distribution configuration "
                "before compute_step."
            )
        return compute_remixmatch_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            num_classes=self.num_classes,
            dist_align_hook=self.dist_align_hook,
            algorithm=self,
            temperature=self.T,
            unsup_warm_up=self.unsup_warm_up,
            mixup_alpha=self.mixup_alpha,
            kl_loss_ratio=self.kl_loss_ratio,
            lambda_u=self.lambda_u,
            supervised_loss_weight=self.supervised_loss_weight,
            iteration=iteration,
            num_train_iter=num_train_iter,
        )

    def _maybe_build_dist_align_hook(self) -> None:
        if self.num_classes is None or self.class_distribution is None:
            return
        if self.class_distribution.numel() != self.num_classes:
            raise ValueError("class_distribution size must match num_classes.")
        self.dist_align_hook = EmaDistributionAlignmentHook(
            num_classes=self.num_classes,
            momentum=self.dist_align_momentum,
            p_target_type="gt",
            p_target=self.class_distribution,
        )
        self.p_model = self.dist_align_hook.p_model
        self.p_target = self.dist_align_hook.p_target


def compute_remixmatch_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Any],
    num_classes: int,
    dist_align_hook: EmaDistributionAlignmentHook,
    iteration: int,
    num_train_iter: int,
    algorithm: ReMixMatchAlgorithm | None = None,
    temperature: float = 0.5,
    unsup_warm_up: float = 1.0 / 64.0,
    mixup_alpha: float = 0.75,
    kl_loss_ratio: float = 0.5,
    lambda_u: float = 1.0,
    supervised_loss_weight: float = 1.0,
) -> QuerySslStepResult:
    """USB `semilearn/algorithms/remixmatch/remixmatch.py::train_step` 핵심."""

    if labeled_batch is None:
        raise ValueError("ReMixMatch requires a labeled_batch.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    num_lb = int(labeled_batch["labels"].shape[0])
    num_ulb = int(unlabeled_batch["weak_input_ids"].shape[0])
    if num_lb <= 0:
        raise ValueError("ReMixMatch labeled batch must not be empty.")
    if num_ulb != num_lb:
        raise ValueError(
            "ReMixMatch requires unlabeled batch size to equal labeled batch size "
            f"(got labeled={num_lb}, unlabeled={num_ulb})."
        )

    with torch.no_grad():
        weak_features = extract_classifier_input_features(
            model,
            input_ids=unlabeled_batch["weak_input_ids"],
            attention_mask=unlabeled_batch["weak_attention_mask"],
        )
        logits_x_ulb_w = classify_classifier_input_features(model, weak_features)
        aligned_prob_x_ulb = dist_align_hook.dist_align(
            probs_x_ulb=compute_prob(logits_x_ulb_w),
            algorithm=algorithm,
        )
        sharpen_prob_x_ulb = sharpen_probabilities(
            aligned_prob_x_ulb,
            temperature=temperature,
        ).detach()

    labeled_features = extract_classifier_input_features(
        model,
        input_ids=labeled_batch["input_ids"],
        attention_mask=labeled_batch["attention_mask"],
    )
    strong_0_features = extract_classifier_input_features(
        model,
        input_ids=unlabeled_batch["strong_0_input_ids"],
        attention_mask=unlabeled_batch["strong_0_attention_mask"],
    )
    strong_1_features = extract_classifier_input_features(
        model,
        input_ids=unlabeled_batch["strong_1_input_ids"],
        attention_mask=unlabeled_batch["strong_1_attention_mask"],
    )
    u1_logits = classify_classifier_input_features(model, strong_0_features)

    input_labels = torch.cat(
        [
            F.one_hot(labeled_batch["labels"], num_classes).to(
                sharpen_prob_x_ulb.dtype
            ),
            sharpen_prob_x_ulb,
            sharpen_prob_x_ulb,
            sharpen_prob_x_ulb,
        ],
        dim=0,
    )
    inputs = torch.cat(
        [
            labeled_features,
            strong_0_features,
            strong_1_features,
            weak_features.detach(),
        ],
        dim=0,
    )
    mixed_x, mixed_y, mixup_lambda = mixup_one_target(
        inputs=inputs,
        targets=input_labels,
        alpha=mixup_alpha,
        bias_toward_primary=True,
    )
    mixed_chunks = list(torch.split(mixed_x, num_lb))
    logits = [
        classify_classifier_input_features(model, mixed_chunk)
        for mixed_chunk in mixed_chunks
    ]
    logits_x = logits[0]
    logits_u = torch.cat(logits[1:], dim=0)

    sup_loss = soft_cross_entropy_loss(
        logits=logits_x,
        targets=mixed_y[:num_lb],
    )
    unsup_loss = probability_mse_loss(
        logits=logits_u,
        targets=mixed_y[num_lb:],
    )
    u1_loss = soft_cross_entropy_loss(
        logits=u1_logits,
        targets=sharpen_prob_x_ulb,
    )
    unsup_warmup = logits_x.new_tensor(
        compute_linear_warmup(
            iteration=iteration,
            warm_up_ratio=unsup_warm_up,
            num_train_iter=num_train_iter,
        )
    )
    effective_lambda_u = logits_x.new_tensor(float(lambda_u)) * unsup_warmup
    effective_lambda_kl = logits_x.new_tensor(float(kl_loss_ratio)) * unsup_warmup
    total_loss = (
        float(supervised_loss_weight) * sup_loss
        + effective_lambda_kl * u1_loss
        + effective_lambda_u * unsup_loss
    )
    return QuerySslStepResult(
        total_loss=total_loss,
        loss_components={
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "u1_loss": u1_loss,
        },
        metrics={
            "unsup_warmup": unsup_warmup,
            "effective_lambda_u": effective_lambda_u.detach(),
            "effective_lambda_kl": effective_lambda_kl.detach(),
            "mixup_lambda": logits_x.new_tensor(mixup_lambda),
        },
        debug_tensors={
            "aligned_prob_x_ulb": aligned_prob_x_ulb.detach(),
            "sharpen_prob_x_ulb": sharpen_prob_x_ulb,
            "mixed_targets": mixed_y.detach(),
        },
    )


def _require_positive_float(value: float, field_name: str) -> float:
    normalized = float(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return normalized


def _require_non_negative_float(value: float, field_name: str) -> float:
    normalized = float(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must not be negative.")
    return normalized


@register_query_ssl_algorithm(
    "remixmatch",
    "re_mixmatch",
    display_name="ReMixMatch",
    required_views=REMIXMATCH_REQUIRED_VIEWS,
    default_uses_labeled_batches=True,
    runtime_requirements=QuerySslRuntimeRequirements(
        batch_surface=QUERY_SSL_BATCH_SURFACE_WEAK_STRONG_PAIR,
        model_outputs=frozenset(
            {
                QUERY_SSL_MODEL_OUTPUT_LOGITS,
                QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES,
            }
        ),
        algorithm_state_surface=frozenset({QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA}),
        optimizer_lifecycle=frozenset({QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP}),
        step_context_required=True,
    ),
)
def build_remixmatch_algorithm(parameters: Mapping[str, Any]) -> ReMixMatchAlgorithm:
    """Hydra method parameter mapping으로 ReMixMatch algorithm을 만든다."""

    return ReMixMatchAlgorithm(
        T=float(parameters.get("T", 0.5)),
        unsup_warm_up=float(parameters.get("unsup_warm_up", 1.0 / 64.0)),
        mixup_alpha=float(parameters.get("mixup_alpha", 0.75)),
        mixup_manifold=bool(parameters.get("mixup_manifold", True)),
        kl_loss_ratio=float(parameters.get("kl_loss_ratio", 0.5)),
        rot_loss_ratio=float(parameters.get("rot_loss_ratio", 0.0)),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
        supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
        dist_align_momentum=float(parameters.get("dist_align_momentum", 0.999)),
    )

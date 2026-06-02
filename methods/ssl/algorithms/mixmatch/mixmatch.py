"""USB MixMatch core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ...base import (
    QUERY_SSL_ALGORITHM_STATE_STATELESS,
    QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
    QUERY_SSL_MODEL_OUTPUT_LOGITS,
    QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP,
    QuerySslRuntimeRequirements,
    QuerySslStepContext,
    QuerySslStepResult,
    TextBatchClassifier,
)
from ...model_capabilities import (
    classify_classifier_input_features,
    extract_classifier_input_features,
)
from ...primitives.losses import probability_mse_loss, soft_cross_entropy_loss
from ...primitives.mixup import mixup_one_target
from ...primitives.probability import compute_prob, sharpen_probabilities
from ...registry import register_query_ssl_algorithm
from ...runtime.schedules import compute_linear_warmup
from ..usb_consistency import (
    USB_MULTIVIEW_REQUIRED_VIEWS,
    validate_usb_consistency_loaders,
)


class MixMatchAlgorithm:
    """MixMatch를 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "mixmatch"

    def __init__(
        self,
        *,
        T: float = 0.5,
        unsup_warm_up: float = 1.0 / 64.0,
        mixup_alpha: float = 0.5,
        mixup_manifold: bool = True,
        lambda_u: float = 1.0,
        supervised_loss_weight: float = 1.0,
    ) -> None:
        self.T = _require_positive_float(T, "T")
        self.unsup_warm_up = float(unsup_warm_up)
        self.mixup_alpha = _require_non_negative_float(mixup_alpha, "mixup_alpha")
        self.mixup_manifold = bool(mixup_manifold)
        if not self.mixup_manifold:
            raise ValueError(
                "TraceMind text MixMatch supports only mixup_manifold=True."
            )
        self.lambda_u = float(lambda_u)
        self.supervised_loss_weight = float(supervised_loss_weight)
        self.num_classes: int | None = None
        self._num_train_iter = 1

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
            algorithm_name="MixMatch",
            train_loader_length=train_loader_length,
            unlabeled_loader_length=unlabeled_loader_length,
            supervised_loss_weight=1.0,
        )

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepResult:
        """context 없는 직접 호출은 첫 step warm-up 의미로 계산한다."""

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
        if self.num_classes is None:
            raise RuntimeError("MixMatch dataset state is not configured.")
        return compute_mixmatch_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            num_classes=self.num_classes,
            temperature=self.T,
            unsup_warm_up=self.unsup_warm_up,
            mixup_alpha=self.mixup_alpha,
            lambda_u=self.lambda_u,
            supervised_loss_weight=self.supervised_loss_weight,
            iteration=iteration,
            num_train_iter=num_train_iter,
        )


def compute_mixmatch_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Any],
    num_classes: int,
    iteration: int,
    num_train_iter: int,
    temperature: float = 0.5,
    unsup_warm_up: float = 1.0 / 64.0,
    mixup_alpha: float = 0.5,
    lambda_u: float = 1.0,
    supervised_loss_weight: float = 1.0,
) -> QuerySslStepResult:
    """USB `semilearn/algorithms/mixmatch/mixmatch.py::train_step` 핵심."""

    if labeled_batch is None:
        raise ValueError("MixMatch requires a labeled_batch.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    num_lb = int(labeled_batch["labels"].shape[0])
    num_ulb = int(unlabeled_batch["weak_input_ids"].shape[0])
    if num_lb <= 0:
        raise ValueError("MixMatch labeled batch must not be empty.")
    if num_ulb != num_lb:
        raise ValueError(
            "MixMatch requires unlabeled batch size to equal labeled batch size "
            f"(got labeled={num_lb}, unlabeled={num_ulb})."
        )

    with torch.no_grad():
        weak_features = extract_classifier_input_features(
            model,
            input_ids=unlabeled_batch["weak_input_ids"],
            attention_mask=unlabeled_batch["weak_attention_mask"],
        )
        strong_features = extract_classifier_input_features(
            model,
            input_ids=unlabeled_batch["strong_input_ids"],
            attention_mask=unlabeled_batch["strong_attention_mask"],
        )
        logits_x_ulb_w = classify_classifier_input_features(model, weak_features)
        logits_x_ulb_s = classify_classifier_input_features(model, strong_features)
        avg_prob_x_ulb = (
            compute_prob(logits_x_ulb_w) + compute_prob(logits_x_ulb_s)
        ) / 2.0
        sharpen_prob_x_ulb = sharpen_probabilities(
            avg_prob_x_ulb,
            temperature=temperature,
        ).detach()

    labeled_features = extract_classifier_input_features(
        model,
        input_ids=labeled_batch["input_ids"],
        attention_mask=labeled_batch["attention_mask"],
    )
    input_labels = torch.cat(
        [
            F.one_hot(labeled_batch["labels"], num_classes).to(
                sharpen_prob_x_ulb.dtype
            ),
            sharpen_prob_x_ulb,
            sharpen_prob_x_ulb,
        ],
        dim=0,
    )
    inputs = torch.cat(
        [
            labeled_features,
            weak_features.detach(),
            strong_features.detach(),
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
    unsup_warmup = logits_x.new_tensor(
        compute_linear_warmup(
            iteration=iteration,
            warm_up_ratio=unsup_warm_up,
            num_train_iter=num_train_iter,
        )
    )
    effective_lambda_u = logits_x.new_tensor(float(lambda_u)) * unsup_warmup
    total_loss = (
        float(supervised_loss_weight) * sup_loss + effective_lambda_u * unsup_loss
    )
    return QuerySslStepResult(
        total_loss=total_loss,
        loss_components={
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
        },
        metrics={
            "unsup_warmup": unsup_warmup,
            "effective_lambda_u": effective_lambda_u.detach(),
            "mixup_lambda": logits_x.new_tensor(mixup_lambda),
        },
        debug_tensors={
            "avg_prob_x_ulb": avg_prob_x_ulb,
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
    "mixmatch",
    display_name="MixMatch",
    required_views=USB_MULTIVIEW_REQUIRED_VIEWS,
    default_uses_labeled_batches=True,
    runtime_requirements=QuerySslRuntimeRequirements(
        batch_surface=QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
        model_outputs=frozenset(
            {
                QUERY_SSL_MODEL_OUTPUT_LOGITS,
                QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES,
            }
        ),
        algorithm_state_surface=frozenset({QUERY_SSL_ALGORITHM_STATE_STATELESS}),
        optimizer_lifecycle=frozenset({QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP}),
        step_context_required=True,
    ),
)
def build_mixmatch_algorithm(parameters: Mapping[str, Any]) -> MixMatchAlgorithm:
    """Hydra method parameter mapping으로 MixMatch algorithm을 만든다."""

    return MixMatchAlgorithm(
        T=float(parameters.get("T", 0.5)),
        unsup_warm_up=float(parameters.get("unsup_warm_up", 1.0 / 64.0)),
        mixup_alpha=float(parameters.get("mixup_alpha", 0.5)),
        mixup_manifold=bool(parameters.get("mixup_manifold", True)),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
        supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
    )

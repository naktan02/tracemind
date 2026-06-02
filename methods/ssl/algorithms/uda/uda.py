"""USB UDA core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ...base import (
    QUERY_SSL_ALGORITHM_STATE_STEP_COUNTER,
    QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
    QuerySslRuntimeRequirements,
    QuerySslStepResult,
    TextBatchClassifier,
)
from ...common import compute_prob
from ...hooks.consistency import CrossEntropyConsistencyLossHook
from ...hooks.masking import FixedThresholdMaskingHook
from ...hooks.pseudo_labeling import HardOrSoftPseudoLabelingHook, PseudoLabelingConfig
from ...registry import register_query_ssl_algorithm
from ...state import (
    build_query_ssl_algorithm_state,
    require_query_ssl_algorithm_state,
)
from ..usb_consistency import (
    USB_MULTIVIEW_REQUIRED_VIEWS,
    compute_unlabeled_weak_strong_logits,
    validate_usb_consistency_loaders,
)


class UDAAlgorithm:
    """UDA를 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "uda"

    def __init__(
        self,
        *,
        temperature: float,
        p_cutoff: float,
        tsa_schedule: str = "none",
        lambda_u: float = 1.0,
    ) -> None:
        self.temperature = float(temperature)
        self.p_cutoff = float(p_cutoff)
        self.tsa_schedule = str(tsa_schedule)
        self.lambda_u = float(lambda_u)
        self._iteration = 0
        self._num_train_iter = 1
        self._num_classes = 1
        self.pseudo_labeling_hook = HardOrSoftPseudoLabelingHook()
        self.masking_hook = FixedThresholdMaskingHook()
        self.consistency_loss_hook = CrossEntropyConsistencyLossHook()

    @property
    def uses_labeled_batches(self) -> bool:
        return True

    def configure_training(self, *, num_train_iter: int) -> None:
        """USB `self.num_train_iter`에 해당하는 TSA denominator를 설정한다."""

        if num_train_iter <= 0:
            raise ValueError("num_train_iter must be positive.")
        self._num_train_iter = int(num_train_iter)
        self._iteration = 0

    def configure_dataset(
        self,
        *,
        num_classes: int,
        unlabeled_row_count: int,
    ) -> None:
        """USB `self.num_classes`에 해당하는 TSA class 수를 설정한다."""

        del unlabeled_row_count
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self._num_classes = int(num_classes)

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        validate_usb_consistency_loaders(
            algorithm_name="UDA",
            train_loader_length=train_loader_length,
            unlabeled_loader_length=unlabeled_loader_length,
            supervised_loss_weight=1.0,
        )

    def export_state(self) -> Mapping[str, Any]:
        """중단 재개용 UDA iteration state를 내보낸다."""

        return build_query_ssl_algorithm_state(
            algorithm_name=self.algorithm_name,
            configured=True,
            metadata={
                "iteration": self._iteration,
                "num_train_iter": self._num_train_iter,
                "num_classes": self._num_classes,
                "tsa_schedule": self.tsa_schedule,
            },
        )

    def load_state(self, state: Mapping[str, Any]) -> None:
        """저장된 UDA iteration state를 복원한다."""

        state = require_query_ssl_algorithm_state(
            state=state,
            algorithm_name=self.algorithm_name,
        )
        self._iteration = int(state.get("iteration", self._iteration))
        self._num_train_iter = int(state.get("num_train_iter", self._num_train_iter))
        self._num_classes = int(state.get("num_classes", self._num_classes))

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepResult:
        output = compute_uda_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            temperature=self.temperature,
            p_cutoff=self.p_cutoff,
            tsa_schedule=self.tsa_schedule,
            iteration=self._iteration,
            num_train_iter=self._num_train_iter,
            num_classes=self._num_classes,
            lambda_u=self.lambda_u,
        )
        self._iteration += 1
        return output


def compute_uda_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Any],
    temperature: float,
    p_cutoff: float,
    tsa_schedule: str,
    iteration: int,
    num_train_iter: int,
    num_classes: int,
    lambda_u: float = 1.0,
) -> QuerySslStepResult:
    """USB `semilearn/algorithms/uda/uda.py::train_step` 핵심."""

    if labeled_batch is None:
        raise ValueError("UDA requires a labeled_batch.")
    if iteration < 0:
        raise ValueError("iteration must not be negative.")
    if num_train_iter <= 0:
        raise ValueError("num_train_iter must be positive.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    logits_x_lb = model(
        input_ids=labeled_batch["input_ids"],
        attention_mask=labeled_batch["attention_mask"],
    )
    logits_x_ulb_s, logits_x_ulb_w = compute_unlabeled_weak_strong_logits(
        model=model,
        unlabeled_batch=unlabeled_batch,
    )

    tsa = logits_x_lb.new_tensor(
        compute_tsa_threshold(
            schedule=tsa_schedule,
            iteration=iteration,
            num_train_iter=num_train_iter,
            num_classes=num_classes,
        )
    )
    sup_mask = torch.max(compute_prob(logits_x_lb), dim=-1)[0].le(tsa).float().detach()
    sup_loss = (
        F.cross_entropy(
            logits_x_lb,
            labeled_batch["labels"],
            reduction="none",
        )
        * sup_mask
    ).mean()

    probs_x_ulb_w = compute_prob(logits_x_ulb_w.detach())
    mask = FixedThresholdMaskingHook().build_mask(
        probs_x_ulb_w=probs_x_ulb_w,
        p_cutoff=p_cutoff,
    )
    pseudo_label = HardOrSoftPseudoLabelingHook().generate_targets(
        probs_x_ulb_w=probs_x_ulb_w,
        config=PseudoLabelingConfig(
            use_hard_label=False,
            temperature=temperature,
        ),
    )
    unsup_loss = CrossEntropyConsistencyLossHook().compute_loss(
        logits=logits_x_ulb_s,
        targets=pseudo_label,
        mask=mask,
    )
    total_loss = sup_loss + float(lambda_u) * unsup_loss
    return QuerySslStepResult(
        total_loss=total_loss,
        loss_components={
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
        },
        metrics={
            "util_ratio": mask.float().mean(),
            "tsa": tsa,
        },
        debug_tensors={
            "mask": mask,
            "sup_mask": sup_mask,
            "pseudo_label": pseudo_label,
        },
    )


def compute_tsa_threshold(
    *,
    schedule: str,
    iteration: int,
    num_train_iter: int,
    num_classes: int,
) -> float:
    """USB UDA `TSA` schedule 수식을 계산한다."""

    if iteration < 0:
        raise ValueError("iteration must not be negative.")
    if num_train_iter <= 0:
        raise ValueError("num_train_iter must be positive.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    normalized_schedule = str(schedule)
    if normalized_schedule == "none":
        return 1.0

    training_progress = float(iteration) / float(num_train_iter)
    if normalized_schedule == "linear":
        threshold = training_progress
    elif normalized_schedule == "exp":
        scale = 5
        threshold = math.exp((training_progress - 1.0) * scale)
    elif normalized_schedule == "log":
        scale = 5
        threshold = 1.0 - math.exp(-training_progress * scale)
    else:
        raise ValueError(
            "UDA tsa_schedule must be one of 'none', 'linear', 'exp', or 'log'."
        )
    return threshold * (1.0 - 1.0 / float(num_classes)) + 1.0 / float(num_classes)


@register_query_ssl_algorithm(
    "uda",
    display_name="UDA",
    required_views=USB_MULTIVIEW_REQUIRED_VIEWS,
    default_uses_labeled_batches=True,
    runtime_requirements=QuerySslRuntimeRequirements(
        batch_surface=QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
        algorithm_state_surface=frozenset({QUERY_SSL_ALGORITHM_STATE_STEP_COUNTER}),
    ),
)
def build_uda_algorithm(parameters: Mapping[str, Any]) -> UDAAlgorithm:
    """Hydra method parameter mapping으로 UDA algorithm을 만든다."""

    return UDAAlgorithm(
        temperature=float(parameters["T"]),
        p_cutoff=float(parameters["p_cutoff"]),
        tsa_schedule=str(parameters.get("tsa_schedule", "none")),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
    )

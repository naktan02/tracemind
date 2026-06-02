"""USB PiModel core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
from ...registry import register_query_ssl_algorithm
from ...state import (
    build_query_ssl_algorithm_state,
    require_query_ssl_algorithm_state,
)
from ..usb_consistency import (
    USB_MULTIVIEW_REQUIRED_VIEWS,
    validate_usb_consistency_loaders,
)


class PiModelAlgorithm:
    """PiModel을 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "pimodel"

    def __init__(
        self,
        *,
        unsup_warm_up: float = 0.4,
        lambda_u: float = 1.0,
    ) -> None:
        self.unsup_warm_up = float(unsup_warm_up)
        self.lambda_u = float(lambda_u)
        self._iteration = 0
        self._num_train_iter = 1

    @property
    def uses_labeled_batches(self) -> bool:
        return True

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
        validate_usb_consistency_loaders(
            algorithm_name="PiModel",
            train_loader_length=train_loader_length,
            unlabeled_loader_length=unlabeled_loader_length,
            supervised_loss_weight=1.0,
        )

    def export_state(self) -> Mapping[str, Any]:
        """중단 재개용 PiModel iteration state를 내보낸다."""

        return build_query_ssl_algorithm_state(
            algorithm_name=self.algorithm_name,
            configured=True,
            metadata={
                "iteration": self._iteration,
                "num_train_iter": self._num_train_iter,
                "unsup_warm_up": self.unsup_warm_up,
            },
        )

    def load_state(self, state: Mapping[str, Any]) -> None:
        """저장된 PiModel iteration state를 복원한다."""

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
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepResult:
        output = compute_pimodel_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            unsup_warm_up=self.unsup_warm_up,
            lambda_u=self.lambda_u,
            iteration=self._iteration,
            num_train_iter=self._num_train_iter,
        )
        self._iteration += 1
        return output


def compute_pimodel_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Any],
    iteration: int,
    num_train_iter: int,
    unsup_warm_up: float = 0.4,
    lambda_u: float = 1.0,
) -> QuerySslStepResult:
    """USB `semilearn/algorithms/pimodel/pimodel.py::train_step` 핵심."""

    if labeled_batch is None:
        raise ValueError("PiModel requires a labeled_batch.")
    if iteration < 0:
        raise ValueError("iteration must not be negative.")
    if num_train_iter <= 0:
        raise ValueError("num_train_iter must be positive.")

    logits_x_lb = model(
        input_ids=labeled_batch["input_ids"],
        attention_mask=labeled_batch["attention_mask"],
    )
    logits_x_ulb_w = model(
        input_ids=unlabeled_batch["weak_input_ids"],
        attention_mask=unlabeled_batch["weak_attention_mask"],
    )
    logits_x_ulb_s = model(
        input_ids=unlabeled_batch["strong_input_ids"],
        attention_mask=unlabeled_batch["strong_attention_mask"],
    )

    sup_loss = F.cross_entropy(logits_x_lb, labeled_batch["labels"], reduction="mean")
    probs_x_ulb_w = compute_prob(logits_x_ulb_w.detach())
    probs_x_ulb_s = compute_prob(logits_x_ulb_s)
    unsup_loss = F.mse_loss(probs_x_ulb_s, probs_x_ulb_w, reduction="mean")
    unsup_warmup = logits_x_lb.new_tensor(
        compute_pimodel_unsup_warmup(
            iteration=iteration,
            unsup_warm_up=unsup_warm_up,
            num_train_iter=num_train_iter,
        )
    )
    total_loss = sup_loss + float(lambda_u) * unsup_loss * unsup_warmup
    return QuerySslStepResult(
        total_loss=total_loss,
        loss_components={
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
        },
        metrics={
            "unsup_warmup": unsup_warmup,
        },
        debug_tensors={
            "probs_x_ulb_w": probs_x_ulb_w,
            "probs_x_ulb_s": probs_x_ulb_s,
        },
    )


def compute_pimodel_unsup_warmup(
    *,
    iteration: int,
    unsup_warm_up: float,
    num_train_iter: int,
) -> float:
    """USB PiModel unsupervised warm-up 수식을 계산한다."""

    if iteration < 0:
        raise ValueError("iteration must not be negative.")
    if num_train_iter <= 0:
        raise ValueError("num_train_iter must be positive.")
    if unsup_warm_up <= 0:
        return 1.0
    denominator = float(unsup_warm_up) * float(num_train_iter)
    return max(0.0, min(1.0, float(iteration) / denominator))


@register_query_ssl_algorithm(
    "pimodel",
    "pi_model",
    display_name="PiModel",
    required_views=USB_MULTIVIEW_REQUIRED_VIEWS,
    default_uses_labeled_batches=True,
    runtime_requirements=QuerySslRuntimeRequirements(
        batch_surface=QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
        algorithm_state_surface=frozenset({QUERY_SSL_ALGORITHM_STATE_STEP_COUNTER}),
    ),
)
def build_pimodel_algorithm(parameters: Mapping[str, Any]) -> PiModelAlgorithm:
    """Hydra method parameter mapping으로 PiModel algorithm을 만든다."""

    return PiModelAlgorithm(
        unsup_warm_up=float(parameters.get("unsup_warm_up", 0.4)),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
    )

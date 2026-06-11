"""USB MeanTeacher core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F

from ...base import (
    QUERY_SSL_ALGORITHM_STATE_STEP_COUNTER,
    QUERY_SSL_ALGORITHM_STATE_TEACHER_EMA,
    QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_POST_STEP_HOOK,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP,
    QUERY_SSL_TEACHER_STATE_EMA_TRAINABLE,
    QuerySslRuntimeRequirements,
    QuerySslStepContext,
    QuerySslStepResult,
    TextBatchClassifier,
)
from ...primitives.probability import compute_prob
from ...registry import register_query_ssl_algorithm
from ...runtime.ema import EmaTrainableParameterTeacher
from ...runtime.schedules import compute_linear_warmup
from ...state import (
    build_query_ssl_algorithm_state,
    require_query_ssl_algorithm_state,
)
from ..usb_consistency import (
    USB_MULTIVIEW_REQUIRED_VIEWS,
    validate_usb_consistency_loaders,
)


class MeanTeacherAlgorithm:
    """MeanTeacher를 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "meanteacher"

    def __init__(
        self,
        *,
        ema_m: float = 0.999,
        unsup_warm_up: float = 0.4,
        lambda_u: float = 1.0,
    ) -> None:
        self.ema_m = float(ema_m)
        self.unsup_warm_up = float(unsup_warm_up)
        self.lambda_u = float(lambda_u)
        self._iteration = 0
        self._num_train_iter = 1
        self._ema_teacher: EmaTrainableParameterTeacher | None = None

    @property
    def uses_labeled_batches(self) -> bool:
        return True

    def configure_training(self, *, num_train_iter: int) -> None:
        """USB `self.num_train_iter`에 해당하는 warm-up denominator를 설정한다."""

        if num_train_iter <= 0:
            raise ValueError("num_train_iter must be positive.")
        self._num_train_iter = int(num_train_iter)
        self._iteration = 0

    def configure_model(self, *, model: nn.Module, device: torch.device) -> None:
        """EMA teacher shadow를 현재 model trainable parameter에서 초기화한다."""

        del device
        self._ema_teacher = EmaTrainableParameterTeacher(
            model=model,
            momentum=self.ema_m,
        )

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        validate_usb_consistency_loaders(
            algorithm_name="MeanTeacher",
            train_loader_length=train_loader_length,
            unlabeled_loader_length=unlabeled_loader_length,
            supervised_loss_weight=1.0,
        )

    def export_state(self) -> Mapping[str, Any]:
        """중단 재개용 MeanTeacher iteration/EMA state를 내보낸다."""

        ema_state = (
            None if self._ema_teacher is None else self._ema_teacher.state_dict()
        )
        return build_query_ssl_algorithm_state(
            algorithm_name=self.algorithm_name,
            configured=self._ema_teacher is not None,
            metadata={
                "iteration": self._iteration,
                "num_train_iter": self._num_train_iter,
                "unsup_warm_up": self.unsup_warm_up,
                "ema_m": self.ema_m,
                "ema_parameter_names": (
                    () if ema_state is None else ema_state.parameter_names
                ),
                "ema_shadow": ({} if ema_state is None else dict(ema_state.shadow)),
            },
        )

    def load_state(self, state: Mapping[str, Any]) -> None:
        """저장된 MeanTeacher iteration/EMA state를 복원한다."""

        state = require_query_ssl_algorithm_state(
            state=state,
            algorithm_name=self.algorithm_name,
        )
        self._iteration = int(state.get("iteration", self._iteration))
        self._num_train_iter = int(state.get("num_train_iter", self._num_train_iter))
        if self._ema_teacher is None:
            if state.get("ema_shadow"):
                raise ValueError(
                    "MeanTeacher EMA state must be loaded after model setup."
                )
            return
        raw_shadow = state.get("ema_shadow", {})
        if not isinstance(raw_shadow, Mapping):
            raise ValueError("MeanTeacher state ema_shadow must be a mapping.")
        parameter_names = tuple(
            str(name) for name in state.get("ema_parameter_names", ())
        )
        self._ema_teacher.load_state_dict(
            shadow=raw_shadow,
            parameter_names=parameter_names or None,
        )

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepResult:
        output = compute_meanteacher_step(
            model=model,
            ema_teacher=self._require_ema_teacher(),
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            unsup_warm_up=self.unsup_warm_up,
            lambda_u=self.lambda_u,
            iteration=self._iteration,
            num_train_iter=self._num_train_iter,
        )
        self._iteration += 1
        return output

    def after_optimizer_step(
        self,
        *,
        model: nn.Module,
        step_context: QuerySslStepContext,
    ) -> None:
        """USB EMAHook.after_train_step에 해당하는 post-step shadow update."""

        del step_context
        self._require_ema_teacher().update(model)

    def _require_ema_teacher(self) -> EmaTrainableParameterTeacher:
        if self._ema_teacher is None:
            raise RuntimeError("MeanTeacher requires configure_model before training.")
        return self._ema_teacher


def compute_meanteacher_step(
    *,
    model: TextBatchClassifier,
    ema_teacher: EmaTrainableParameterTeacher,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Any],
    iteration: int,
    num_train_iter: int,
    unsup_warm_up: float = 0.4,
    lambda_u: float = 1.0,
) -> QuerySslStepResult:
    """USB `semilearn/algorithms/meanteacher/meanteacher.py::train_step` 핵심."""

    if labeled_batch is None:
        raise ValueError("MeanTeacher requires a labeled_batch.")
    if not isinstance(model, nn.Module):
        raise TypeError("MeanTeacher model must be an nn.Module for EMA shadow swap.")

    logits_x_lb = model(
        input_ids=labeled_batch["input_ids"],
        attention_mask=labeled_batch["attention_mask"],
    )
    with torch.no_grad():
        logits_x_ulb_w = functional_call(
            model,
            ema_teacher.shadow_parameter_mapping(model),
            args=(),
            kwargs={
                "input_ids": unlabeled_batch["weak_input_ids"],
                "attention_mask": unlabeled_batch["weak_attention_mask"],
            },
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
        compute_linear_warmup(
            iteration=iteration,
            warm_up_ratio=unsup_warm_up,
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


@register_query_ssl_algorithm(
    "meanteacher",
    "mean_teacher",
    display_name="MeanTeacher",
    required_views=USB_MULTIVIEW_REQUIRED_VIEWS,
    default_uses_labeled_batches=True,
    runtime_requirements=QuerySslRuntimeRequirements(
        batch_surface=QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
        algorithm_state_surface=frozenset(
            {
                QUERY_SSL_ALGORITHM_STATE_STEP_COUNTER,
                QUERY_SSL_ALGORITHM_STATE_TEACHER_EMA,
            }
        ),
        optimizer_lifecycle=frozenset(
            {
                QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP,
                QUERY_SSL_OPTIMIZER_LIFECYCLE_POST_STEP_HOOK,
            }
        ),
        teacher_state=QUERY_SSL_TEACHER_STATE_EMA_TRAINABLE,
    ),
)
def build_meanteacher_algorithm(parameters: Mapping[str, Any]) -> MeanTeacherAlgorithm:
    """Hydra method parameter mapping으로 MeanTeacher algorithm을 만든다."""

    return MeanTeacherAlgorithm(
        ema_m=float(parameters.get("ema_m", 0.999)),
        unsup_warm_up=float(parameters.get("unsup_warm_up", 0.4)),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
    )

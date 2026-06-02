"""USB Dash core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ...base import (
    QUERY_SSL_ALGORITHM_STATE_ADAPTIVE_THRESHOLD,
    QUERY_SSL_ALGORITHM_STATE_STEP_COUNTER,
    QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
    QUERY_SSL_INPUT_TRANSFORM_NONE,
    QUERY_SSL_MODEL_OUTPUT_LOGITS,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP,
    QUERY_SSL_TEACHER_STATE_NONE,
    QuerySslRuntimeRequirements,
    QuerySslStepContext,
    QuerySslStepResult,
    TextBatchClassifier,
)
from ...common import compute_prob
from ...hooks.consistency import consistency_cross_entropy_loss
from ...hooks.pseudo_labeling import (
    HardOrSoftPseudoLabelingHook,
    PseudoLabelingConfig,
)
from ...hooks.supervised import compute_labeled_cross_entropy_loss
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


@dataclass(slots=True)
class DashThresholdState:
    """Dash dynamic threshold의 현재 상태."""

    rho_init: float | None
    rho: float | None
    rho_update_cnt: int
    use_hard_label: bool


class DashThresholdingHook:
    """USB `DashThresholdingHook`의 dynamic threshold mask."""

    hook_name: str = "dash_thresholding"

    def __init__(
        self,
        *,
        rho_min: float,
        gamma: float,
        C: float,
    ) -> None:
        if float(rho_min) <= 0:
            raise ValueError("rho_min must be positive.")
        if float(gamma) <= 0:
            raise ValueError("gamma must be positive.")
        if float(C) <= 0:
            raise ValueError("C must be positive.")
        self.rho_min = float(rho_min)
        self.gamma = float(gamma)
        self.C = float(C)
        self.state = DashThresholdState(
            rho_init=None,
            rho=None,
            rho_update_cnt=0,
            use_hard_label=False,
        )

    @property
    def needs_initial_selection_loss(self) -> bool:
        return self.state.rho_init is None

    def configure_initial_selection_loss(self, *, selection_loss: float) -> None:
        """USB warm-up eval/loss에 해당하는 rho_init 값을 설정한다."""

        if float(selection_loss) <= 0:
            raise ValueError("Dash initial selection loss must be positive.")
        if self.state.rho_init is None:
            self.state.rho_init = float(selection_loss)
        if self.state.rho is None:
            self.state.rho = float(selection_loss)

    def export_state(self) -> dict[str, object]:
        return {
            "rho_init": self.state.rho_init,
            "rho": self.state.rho,
            "rho_update_cnt": self.state.rho_update_cnt,
            "use_hard_label": self.state.use_hard_label,
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        rho_init = state.get("rho_init")
        rho = state.get("rho")
        self.state = DashThresholdState(
            rho_init=None if rho_init is None else float(rho_init),
            rho=None if rho is None else float(rho),
            rho_update_cnt=int(state.get("rho_update_cnt", 0)),
            use_hard_label=bool(state.get("use_hard_label", False)),
        )

    @torch.no_grad()
    def build_mask(
        self,
        *,
        logits_x_ulb_w: Tensor,
        temperature: float,
        step_context: QuerySslStepContext,
    ) -> Tensor:
        self._update(step_context)
        if self.state.rho is None:
            raise RuntimeError("Dash rho must be initialized before masking.")
        if self.state.use_hard_label:
            pseudo_label = torch.argmax(logits_x_ulb_w, dim=-1).detach()
        else:
            pseudo_label = compute_prob(logits_x_ulb_w.detach() / float(temperature))
        loss_w = _per_sample_cross_entropy(
            logits=logits_x_ulb_w,
            targets=pseudo_label,
        )
        return loss_w.le(float(self.state.rho)).to(logits_x_ulb_w.dtype)

    def _update(self, step_context: QuerySslStepContext) -> None:
        if self.state.rho_init is None:
            raise RuntimeError("Dash rho_init must be configured before training.")

        should_update = (
            self.state.rho is None
            or step_context.step_index == 1
            and (step_context.epoch_index - 1) % 10 == 0
        )
        if should_update:
            rho = (
                self.C
                * (self.gamma ** (-self.state.rho_update_cnt))
                * self.state.rho_init
            )
            self.state.rho = max(float(rho), self.rho_min)
            self.state.rho_update_cnt += 1

        self.state.use_hard_label = bool(self.state.rho <= self.rho_min)


class DashAlgorithm:
    """Dash를 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "dash"

    def __init__(
        self,
        *,
        T: float = 0.5,
        gamma: float = 1.27,
        C: float = 1.0001,
        rho_min: float = 0.05,
        lambda_u: float = 1.0,
        supervised_loss_weight: float = 1.0,
        rho_init: float | None = None,
        thresholding_hook: DashThresholdingHook | None = None,
        pseudo_labeling_hook: HardOrSoftPseudoLabelingHook | None = None,
    ) -> None:
        self.T = float(T)
        self.gamma = float(gamma)
        self.C = float(C)
        self.rho_min = float(rho_min)
        self.lambda_u = float(lambda_u)
        self.supervised_loss_weight = float(supervised_loss_weight)
        self.thresholding_hook = thresholding_hook or DashThresholdingHook(
            rho_min=self.rho_min,
            gamma=self.gamma,
            C=self.C,
        )
        self.pseudo_labeling_hook = (
            pseudo_labeling_hook or HardOrSoftPseudoLabelingHook()
        )
        if rho_init is not None:
            self.thresholding_hook.configure_initial_selection_loss(
                selection_loss=float(rho_init),
            )

    @property
    def uses_labeled_batches(self) -> bool:
        return self.supervised_loss_weight > 0

    @property
    def needs_initial_selection_loss(self) -> bool:
        return self.thresholding_hook.needs_initial_selection_loss

    def configure_initial_selection_loss(self, *, selection_loss: float) -> None:
        self.thresholding_hook.configure_initial_selection_loss(
            selection_loss=selection_loss,
        )

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        validate_usb_consistency_loaders(
            algorithm_name="Dash",
            train_loader_length=train_loader_length,
            unlabeled_loader_length=unlabeled_loader_length,
            supervised_loss_weight=self.supervised_loss_weight,
        )

    def export_state(self) -> Mapping[str, Any]:
        return build_query_ssl_algorithm_state(
            algorithm_name=self.algorithm_name,
            configured=not self.needs_initial_selection_loss,
            metadata={
                **self.thresholding_hook.export_state(),
            },
        )

    def load_state(self, state: Mapping[str, Any]) -> None:
        state = require_query_ssl_algorithm_state(
            state=state,
            algorithm_name=self.algorithm_name,
        )
        self.thresholding_hook.load_state(state)

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepResult:
        raise RuntimeError("Dash requires QuerySslStepContext.")

    def compute_step_with_context(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
        step_context: QuerySslStepContext,
    ) -> QuerySslStepResult:
        return compute_dash_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            step_context=step_context,
            temperature=self.T,
            lambda_u=self.lambda_u,
            supervised_loss_weight=self.supervised_loss_weight,
            thresholding_hook=self.thresholding_hook,
            pseudo_labeling_hook=self.pseudo_labeling_hook,
        )


def compute_dash_step(
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Any],
    step_context: QuerySslStepContext,
    temperature: float,
    lambda_u: float = 1.0,
    supervised_loss_weight: float = 1.0,
    thresholding_hook: DashThresholdingHook,
    pseudo_labeling_hook: HardOrSoftPseudoLabelingHook | None = None,
) -> QuerySslStepResult:
    """USB `semilearn/algorithms/dash/dash.py::train_step` 핵심."""

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

    mask = thresholding_hook.build_mask(
        logits_x_ulb_w=logits_x_ulb_w,
        temperature=temperature,
        step_context=step_context,
    )
    effective_pseudo_labeling_hook = (
        pseudo_labeling_hook or HardOrSoftPseudoLabelingHook()
    )
    pseudo_label = effective_pseudo_labeling_hook.generate_targets_from_logits(
        logits_x_ulb_w=logits_x_ulb_w,
        config=PseudoLabelingConfig(
            use_hard_label=thresholding_hook.state.use_hard_label,
            temperature=temperature,
        ),
    )
    unsup_loss = consistency_cross_entropy_loss(
        logits=logits_x_ulb_s,
        targets=pseudo_label,
        mask=mask,
    )
    rho = logits_x_ulb_s.new_tensor(float(thresholding_hook.state.rho or 0.0))
    total_loss = supervised_loss_weight * sup_loss + float(lambda_u) * unsup_loss
    return QuerySslStepResult(
        total_loss=total_loss,
        loss_components={
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
        },
        metrics={
            "util_ratio": mask.float().mean(),
            "rho": rho,
            "rho_update_cnt": logits_x_ulb_s.new_tensor(
                float(thresholding_hook.state.rho_update_cnt)
            ),
            "use_hard_label": logits_x_ulb_s.new_tensor(
                float(thresholding_hook.state.use_hard_label)
            ),
        },
        debug_tensors={
            "mask": mask,
            "pseudo_label": pseudo_label,
        },
    )


def _per_sample_cross_entropy(*, logits: Tensor, targets: Tensor) -> Tensor:
    if targets.dtype in (torch.int32, torch.int64, torch.long):
        return F.cross_entropy(logits, targets, reduction="none")
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1)


@register_query_ssl_algorithm(
    "dash",
    display_name="Dash",
    required_views=USB_MULTIVIEW_REQUIRED_VIEWS,
    default_uses_labeled_batches=True,
    runtime_requirements=QuerySslRuntimeRequirements(
        batch_surface=QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
        model_outputs=frozenset({QUERY_SSL_MODEL_OUTPUT_LOGITS}),
        algorithm_state_surface=frozenset(
            {
                QUERY_SSL_ALGORITHM_STATE_ADAPTIVE_THRESHOLD,
                QUERY_SSL_ALGORITHM_STATE_STEP_COUNTER,
            }
        ),
        input_transform_surface=QUERY_SSL_INPUT_TRANSFORM_NONE,
        optimizer_lifecycle=frozenset({QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP}),
        teacher_state=QUERY_SSL_TEACHER_STATE_NONE,
        step_context_required=True,
    ),
)
def build_dash_algorithm(parameters: Mapping[str, Any]) -> DashAlgorithm:
    """Hydra method parameter mapping으로 Dash algorithm을 만든다."""

    raw_rho_init = parameters.get("rho_init")
    return DashAlgorithm(
        T=float(parameters.get("T", 0.5)),
        gamma=float(parameters.get("gamma", 1.27)),
        C=float(parameters.get("C", 1.0001)),
        rho_min=float(parameters.get("rho_min", 0.05)),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
        supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
        rho_init=None if raw_rho_init is None else float(raw_rho_init),
    )

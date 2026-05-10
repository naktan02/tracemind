"""USB FlexMatch core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ...base import QuerySslRequiredViews, QuerySslStepOutput, TextBatchClassifier
from ...common import compute_prob
from ...hooks.consistency import CrossEntropyConsistencyLossHook
from ...hooks.masking import FixedThresholdMaskingHook
from ...hooks.objective import SslObjectiveHooks
from ...hooks.pseudo_labeling import (
    HardOrSoftPseudoLabelingHook,
    PseudoLabelingConfig,
)
from ...registry import register_query_ssl_algorithm


class FlexMatchThresholdingHook:
    """USB `FlexMatchThresholdingHook`의 adaptive thresholding state."""

    hook_name: str = "flexmatch_thresholding"

    def __init__(
        self,
        *,
        ulb_dest_len: int,
        num_classes: int,
        thresh_warmup: bool = True,
    ) -> None:
        if ulb_dest_len <= 0:
            raise ValueError("ulb_dest_len must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.ulb_dest_len = int(ulb_dest_len)
        self.num_classes = int(num_classes)
        self.thresh_warmup = bool(thresh_warmup)
        self.selected_label = torch.ones((self.ulb_dest_len,), dtype=torch.long) * -1
        self.classwise_acc = torch.zeros((self.num_classes,))

    @torch.no_grad()
    def update(self, *args, **kwargs) -> None:
        """USB FlexMatch utils.py `update`와 같은 classwise_acc 갱신."""

        del args, kwargs
        pseudo_counter = Counter(self.selected_label.tolist())
        if max(pseudo_counter.values()) < self.ulb_dest_len:
            if self.thresh_warmup:
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(
                        pseudo_counter.values()
                    )
            else:
                wo_negative_one = deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                if not wo_negative_one:
                    return
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(
                        wo_negative_one.values()
                    )

    @torch.no_grad()
    def masking(
        self,
        algorithm: FlexMatchAlgorithm,
        logits_x_ulb: Tensor,
        idx_ulb: Tensor,
        softmax_x_ulb: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        """USB FlexMatch utils.py `masking`의 convex threshold 수식."""

        del args, kwargs
        if not self.selected_label.is_cuda and logits_x_ulb.is_cuda:
            self.selected_label = self.selected_label.to(logits_x_ulb.device)
        if not self.classwise_acc.is_cuda and logits_x_ulb.is_cuda:
            self.classwise_acc = self.classwise_acc.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = compute_prob(logits_x_ulb.detach())
        else:
            probs_x_ulb = logits_x_ulb.detach()
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1)
        mask = max_probs.ge(
            algorithm.p_cutoff
            * (self.classwise_acc[max_idx] / (2.0 - self.classwise_acc[max_idx]))
        )
        select = max_probs.ge(algorithm.p_cutoff)
        mask = mask.to(max_probs.dtype)

        if idx_ulb[select == 1].nelement() != 0:
            self.selected_label[idx_ulb[select == 1]] = max_idx[select == 1]
        self.update()

        return mask


class FlexMatchStepOutput:
    """FlexMatch 한 step의 loss/diagnostics 결과."""

    def __init__(
        self,
        *,
        total_loss: Tensor,
        sup_loss: Tensor,
        unsup_loss: Tensor,
        mask: Tensor,
        classwise_acc: Tensor,
    ) -> None:
        self.total_loss = total_loss
        self.sup_loss = sup_loss
        self.unsup_loss = unsup_loss
        self.mask = mask
        self.classwise_acc = classwise_acc

    @property
    def util_ratio(self) -> Tensor:
        return self.mask.float().mean()

    @property
    def loss_components(self) -> dict[str, Tensor]:
        return {
            "sup_loss": self.sup_loss,
            "unsup_loss": self.unsup_loss,
        }

    @property
    def metrics(self) -> dict[str, Tensor]:
        return {"util_ratio": self.util_ratio}


def build_flexmatch_objective_hooks() -> SslObjectiveHooks:
    """FlexMatch baseline의 tensor-level hook 조합을 만든다."""

    return SslObjectiveHooks(
        pseudo_labeling=HardOrSoftPseudoLabelingHook(),
        masking=FixedThresholdMaskingHook(),
        consistency_loss=CrossEntropyConsistencyLossHook(),
    )


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
        hooks: SslObjectiveHooks | None = None,
    ) -> None:
        self.temperature = float(temperature)
        self.p_cutoff = float(p_cutoff)
        self.hard_label = bool(hard_label)
        self.thresh_warmup = bool(thresh_warmup)
        self.lambda_u = float(lambda_u)
        self.supervised_loss_weight = float(supervised_loss_weight)
        self.hooks = hooks or build_flexmatch_objective_hooks()
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
        """USB `ulb_dest_len`/`num_classes` 기반 masking hook state를 초기화한다."""

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
        if unlabeled_loader_length == 0:
            raise ValueError("FlexMatch unlabeled_loader must not be empty.")
        if self.supervised_loss_weight > 0 and train_loader_length == 0:
            raise ValueError(
                "FlexMatch labeled train_loader must not be empty when "
                "supervised_loss_weight > 0."
            )

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepOutput:
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
            hooks=self.hooks,
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
    hooks: SslObjectiveHooks | None = None,
    masking_hook: FlexMatchThresholdingHook,
    algorithm: FlexMatchAlgorithm | None = None,
) -> FlexMatchStepOutput:
    """USB `semilearn/algorithms/flexmatch/flexmatch.py::train_step` 핵심."""

    if labeled_batch is None:
        sup_loss = None
    else:
        logits_x_lb = model(
            input_ids=labeled_batch["input_ids"],
            attention_mask=labeled_batch["attention_mask"],
        )
        sup_loss = F.cross_entropy(
            logits_x_lb, labeled_batch["labels"], reduction="mean"
        )

    logits_x_ulb_s = model(
        input_ids=unlabeled_batch["strong_input_ids"],
        attention_mask=unlabeled_batch["strong_attention_mask"],
    )
    with torch.no_grad():
        logits_x_ulb_w = model(
            input_ids=unlabeled_batch["weak_input_ids"],
            attention_mask=unlabeled_batch["weak_attention_mask"],
        )
    if sup_loss is None:
        sup_loss = logits_x_ulb_s.new_zeros(())

    probs_x_ulb_w = compute_prob(logits_x_ulb_w.detach())
    effective_hooks = hooks or build_flexmatch_objective_hooks()
    masking_algorithm = algorithm or _FlexMatchMaskingAlgorithm(p_cutoff=p_cutoff)
    mask = masking_hook.masking(
        masking_algorithm,
        logits_x_ulb=probs_x_ulb_w,
        softmax_x_ulb=False,
        idx_ulb=_require_row_indices(unlabeled_batch).to(probs_x_ulb_w.device),
    )
    pseudo_label = effective_hooks.pseudo_labeling.generate_targets(
        probs_x_ulb_w=probs_x_ulb_w,
        config=PseudoLabelingConfig(
            use_hard_label=hard_label,
            temperature=temperature,
        ),
    )
    unsup_loss = effective_hooks.consistency_loss.compute_loss(
        logits=logits_x_ulb_s,
        targets=pseudo_label,
        mask=mask,
    )
    total_loss = supervised_loss_weight * sup_loss + lambda_u * unsup_loss
    return FlexMatchStepOutput(
        total_loss=total_loss,
        sup_loss=sup_loss,
        unsup_loss=unsup_loss,
        mask=mask,
        classwise_acc=masking_hook.classwise_acc,
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
    required_views=QuerySslRequiredViews(
        view_names=("text", "aug_0", "aug_1"),
        view_builder_name="usb_multiview",
    ),
    default_uses_labeled_batches=True,
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

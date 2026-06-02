"""SoftMatch method-local sample weighting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from methods.ssl.primitives.probability import compute_prob


class SoftMatchWeightingHook:
    """USB SoftMatch learnable truncated Gaussian weighting."""

    hook_name: str = "softmatch_weighting"

    def __init__(
        self,
        *,
        num_classes: int,
        n_sigma: float = 2.0,
        momentum: float = 0.999,
        per_class: bool = False,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if n_sigma <= 0:
            raise ValueError("n_sigma must be positive.")
        self.num_classes = int(num_classes)
        self.n_sigma = float(n_sigma)
        self.m = float(momentum)
        self.per_class = bool(per_class)
        if self.per_class:
            self.prob_max_mu_t = torch.ones((self.num_classes,)) / self.num_classes
            self.prob_max_var_t = torch.ones((self.num_classes,))
        else:
            self.prob_max_mu_t = torch.tensor(1.0 / self.num_classes)
            self.prob_max_var_t = torch.tensor(1.0)

    @torch.no_grad()
    def update(self, probs_x_ulb: Tensor) -> tuple[Tensor, Tensor]:
        """max probability EMA mean/variance state를 갱신한다."""

        self._move_state_to_device(probs_x_ulb.device)
        max_probs, max_idx = probs_x_ulb.detach().max(dim=-1)
        if not self.per_class:
            prob_max_mu_t = max_probs.mean()
            prob_max_var_t = _safe_variance(max_probs)
            self.prob_max_mu_t = (
                self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
            )
            self.prob_max_var_t = (
                self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t
            )
            return max_probs, max_idx

        prob_max_mu_t = torch.zeros_like(self.prob_max_mu_t)
        prob_max_var_t = torch.ones_like(self.prob_max_var_t)
        for class_index in range(self.num_classes):
            class_probs = max_probs[max_idx == class_index]
            if len(class_probs) > 1:
                prob_max_mu_t[class_index] = class_probs.mean()
                prob_max_var_t[class_index] = _safe_variance(class_probs)
        self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
        self.prob_max_var_t = (
            self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t
        )
        return max_probs, max_idx

    @torch.no_grad()
    def masking(
        self,
        *,
        logits_x_ulb: Tensor,
        softmax_x_ulb: bool = True,
    ) -> Tensor:
        """SoftMatch sample weight mask를 계산한다."""

        self._move_state_to_device(logits_x_ulb.device)
        probs_x_ulb = (
            compute_prob(logits_x_ulb.detach())
            if softmax_x_ulb
            else logits_x_ulb.detach()
        )
        max_probs, max_idx = self.update(probs_x_ulb)
        if not self.per_class:
            mu = self.prob_max_mu_t
            var = self.prob_max_var_t
        else:
            mu = self.prob_max_mu_t[max_idx]
            var = self.prob_max_var_t[max_idx]
        denominator = (2 * var.clamp(min=1e-12)) / (self.n_sigma**2)
        return torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / denominator))

    def export_state(self) -> dict[str, Any]:
        """resume checkpoint에 넣을 SoftMatch weighting state를 반환한다."""

        return {
            "num_classes": self.num_classes,
            "n_sigma": self.n_sigma,
            "momentum": self.m,
            "per_class": self.per_class,
            "prob_max_mu_t": self.prob_max_mu_t.detach().cpu().clone(),
            "prob_max_var_t": self.prob_max_var_t.detach().cpu().clone(),
        }

    def load_state(
        self,
        state: Mapping[str, Any],
        *,
        device: torch.device | None = None,
    ) -> None:
        """저장된 SoftMatch weighting state를 복원한다."""

        if int(state.get("num_classes", self.num_classes)) != self.num_classes:
            raise ValueError("SoftMatch weighting num_classes mismatch.")
        if bool(state.get("per_class", self.per_class)) != self.per_class:
            raise ValueError("SoftMatch weighting per_class mismatch.")
        target_device = device or self.prob_max_mu_t.device
        expected_shape = (self.num_classes,) if self.per_class else ()
        self.prob_max_mu_t = _require_tensor_state(
            state,
            "prob_max_mu_t",
            expected_shape=expected_shape,
            device=target_device,
        )
        self.prob_max_var_t = _require_tensor_state(
            state,
            "prob_max_var_t",
            expected_shape=expected_shape,
            device=target_device,
        )

    def _move_state_to_device(self, device: torch.device) -> None:
        if self.prob_max_mu_t.device != device:
            self.prob_max_mu_t = self.prob_max_mu_t.to(device)
        if self.prob_max_var_t.device != device:
            self.prob_max_var_t = self.prob_max_var_t.to(device)


def _safe_variance(values: Tensor) -> Tensor:
    if values.numel() <= 1:
        return values.new_zeros(())
    return torch.var(values, unbiased=True)


def _require_tensor_state(
    state: Mapping[str, Any],
    field_name: str,
    *,
    expected_shape: tuple[int, ...],
    device: torch.device,
) -> Tensor:
    value = state.get(field_name)
    if not isinstance(value, Tensor):
        raise ValueError(f"SoftMatch weighting state requires tensor {field_name}.")
    tensor = value.detach().clone().to(device)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"SoftMatch weighting state {field_name} shape mismatch.")
    return tensor

"""SSL distribution alignment hooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch
from torch import Tensor


class DistributionAlignmentStateReceiver(Protocol):
    """Distribution alignment hook이 algorithm-local mirror state를 갱신하는 surface."""

    p_model: Tensor | None
    p_target: Tensor | None


class EmaDistributionAlignmentHook:
    """USB `DistAlignEMAHook`의 non-distributed EMA state."""

    hook_name: str = "ema_distribution_alignment"

    def __init__(
        self,
        *,
        num_classes: int,
        momentum: float = 0.999,
        p_target_type: str = "uniform",
        p_target: Tensor | None = None,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = int(num_classes)
        self.m = float(momentum)
        self.p_target_type = _normalize_p_target_type(p_target_type)
        self.update_p_target, self.p_target = self._build_p_target(
            p_target_type=self.p_target_type,
            p_target=p_target,
        )
        self.p_model: Tensor | None = None

    @torch.no_grad()
    def dist_align(
        self,
        *,
        probs_x_ulb: Tensor,
        probs_x_lb: Tensor | None = None,
        algorithm: DistributionAlignmentStateReceiver | None = None,
    ) -> Tensor:
        """EMA distribution alignment를 적용한 unlabeled probability."""

        self._move_state_to_device(probs_x_ulb.device)
        self.update_p(probs_x_ulb=probs_x_ulb, probs_x_lb=probs_x_lb)

        if self.p_model is None:  # pragma: no cover - defensive
            raise RuntimeError("EMA distribution alignment state was not updated.")

        aligned_probs = probs_x_ulb * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        aligned_probs = aligned_probs / aligned_probs.sum(dim=-1, keepdim=True)

        if algorithm is not None:
            algorithm.p_model = self.p_model
            algorithm.p_target = self.p_target
        return aligned_probs

    @torch.no_grad()
    def update_p(
        self,
        *,
        probs_x_ulb: Tensor,
        probs_x_lb: Tensor | None = None,
    ) -> None:
        """USB `DistAlignEMAHook.update_p`의 non-distributed EMA update."""

        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model is None:
            self.p_model = probs_x_ulb.mean(dim=0)
        else:
            self.p_model = self.p_model * self.m + probs_x_ulb.mean(dim=0) * (
                1 - self.m
            )
        if self.update_p_target:
            if probs_x_lb is None:
                raise ValueError(
                    "model target distribution alignment requires probs_x_lb."
                )
            self.p_target = self.p_target * self.m + probs_x_lb.mean(dim=0) * (
                1 - self.m
            )

    def _build_p_target(
        self,
        *,
        p_target_type: str,
        p_target: Tensor | None,
    ) -> tuple[bool, Tensor]:
        if p_target_type == "uniform":
            return False, torch.ones((self.num_classes,)) / self.num_classes
        if p_target_type == "model":
            return True, torch.ones((self.num_classes,)) / self.num_classes
        if p_target is None:
            raise ValueError("gt EMA distribution alignment requires p_target.")
        normalized = p_target.detach().to(dtype=torch.float32).reshape(-1)
        if normalized.numel() != self.num_classes:
            raise ValueError("p_target size must match num_classes.")
        return False, normalized

    def _move_state_to_device(self, device: torch.device) -> None:
        if self.p_target.device != device:
            self.p_target = self.p_target.to(device)
        if self.p_model is not None and self.p_model.device != device:
            self.p_model = self.p_model.to(device)


class QueueDistributionAlignmentHook:
    """USB `DistAlignQueueHook`의 non-distributed queue 기반 DA state."""

    hook_name: str = "queue_distribution_alignment"

    def __init__(
        self,
        *,
        num_classes: int,
        queue_length: int,
        p_target_type: str = "uniform",
        p_target: Tensor | None = None,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if queue_length <= 0:
            raise ValueError("queue_length must be positive.")
        self.num_classes = int(num_classes)
        self.queue_length = int(queue_length)
        self.p_target_type = _normalize_p_target_type(p_target_type)
        self.p_target_ptr, self.p_target = self._build_p_target(
            p_target_type=self.p_target_type,
            p_target=p_target,
        )
        self.p_model = torch.zeros(
            (self.queue_length, self.num_classes),
            dtype=torch.float32,
        )
        self.p_model_ptr = torch.zeros(1, dtype=torch.long)

    @torch.no_grad()
    def dist_align(
        self,
        *,
        probs_x_ulb: Tensor,
        probs_x_lb: Tensor | None = None,
    ) -> Tensor:
        """queue 평균 분포로 unlabeled probability를 alignment한다."""

        self._move_state_to_device(probs_x_ulb.device)
        self.update_p(probs_x_ulb=probs_x_ulb, probs_x_lb=probs_x_lb)
        p_model = self.p_model.mean(dim=0)
        p_target = self.p_target.mean(dim=0)
        aligned_probs = probs_x_ulb * (p_target + 1e-6) / (p_model + 1e-6)
        return aligned_probs / aligned_probs.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def update_p(
        self,
        *,
        probs_x_ulb: Tensor,
        probs_x_lb: Tensor | None = None,
    ) -> None:
        """현재 batch 평균을 model/target queue에 기록한다."""

        self._move_state_to_device(probs_x_ulb.device)
        probs_x_ulb = probs_x_ulb.detach()
        p_model_ptr = int(self.p_model_ptr.item())
        self.p_model[p_model_ptr] = probs_x_ulb.mean(dim=0)
        self.p_model_ptr[0] = (p_model_ptr + 1) % self.queue_length
        if self.p_target_ptr is None:
            return
        if probs_x_lb is None:
            raise ValueError("model target distribution alignment requires probs_x_lb.")
        p_target_ptr = int(self.p_target_ptr.item())
        self.p_target[p_target_ptr] = probs_x_lb.detach().mean(dim=0)
        self.p_target_ptr[0] = (p_target_ptr + 1) % self.queue_length

    def export_state(self) -> dict[str, Any]:
        """resume checkpoint에 넣을 queue DA state를 반환한다."""

        return {
            "num_classes": self.num_classes,
            "queue_length": self.queue_length,
            "p_target_type": self.p_target_type,
            "p_model": self.p_model.detach().cpu().clone(),
            "p_model_ptr": self.p_model_ptr.detach().cpu().clone(),
            "p_target": self.p_target.detach().cpu().clone(),
            "p_target_ptr": None
            if self.p_target_ptr is None
            else self.p_target_ptr.detach().cpu().clone(),
        }

    def load_state(
        self,
        state: Mapping[str, Any],
        *,
        device: torch.device | None = None,
    ) -> None:
        """저장된 queue DA state를 현재 hook에 복원한다."""

        if int(state.get("num_classes", self.num_classes)) != self.num_classes:
            raise ValueError("Queue DA num_classes does not match current hook.")
        if int(state.get("queue_length", self.queue_length)) != self.queue_length:
            raise ValueError("Queue DA queue_length does not match current hook.")
        target_type = str(state.get("p_target_type", self.p_target_type))
        if target_type != self.p_target_type:
            raise ValueError("Queue DA p_target_type does not match current hook.")
        target_device = device or self.p_model.device
        self.p_model = _require_tensor_state(
            state,
            "p_model",
            expected_shape=(self.queue_length, self.num_classes),
            device=target_device,
        )
        self.p_model_ptr = _require_tensor_state(
            state,
            "p_model_ptr",
            expected_shape=(1,),
            device=target_device,
            dtype=torch.long,
        )
        self.p_target = _require_tensor_state(
            state,
            "p_target",
            expected_shape=(self.queue_length, self.num_classes),
            device=target_device,
        )
        p_target_ptr = state.get("p_target_ptr")
        self.p_target_ptr = (
            None
            if p_target_ptr is None
            else _require_tensor_state(
                state,
                "p_target_ptr",
                expected_shape=(1,),
                device=target_device,
                dtype=torch.long,
            )
        )

    def _build_p_target(
        self,
        *,
        p_target_type: str,
        p_target: Tensor | None,
    ) -> tuple[Tensor | None, Tensor]:
        if p_target_type == "uniform":
            return (
                None,
                torch.ones(
                    (self.queue_length, self.num_classes),
                    dtype=torch.float32,
                )
                / self.num_classes,
            )
        if p_target_type == "model":
            return (
                torch.zeros(1, dtype=torch.long),
                torch.zeros(
                    (self.queue_length, self.num_classes),
                    dtype=torch.float32,
                ),
            )
        if p_target is None:
            raise ValueError("gt queue distribution alignment requires p_target.")
        normalized = p_target.detach().to(dtype=torch.float32).reshape(-1)
        if normalized.numel() != self.num_classes:
            raise ValueError("p_target size must match num_classes.")
        return None, normalized.unsqueeze(0).repeat((self.queue_length, 1))

    def _move_state_to_device(self, device: torch.device) -> None:
        if self.p_model.device != device:
            self.p_model = self.p_model.to(device)
        if self.p_model_ptr.device != device:
            self.p_model_ptr = self.p_model_ptr.to(device)
        if self.p_target.device != device:
            self.p_target = self.p_target.to(device)
        if self.p_target_ptr is not None and self.p_target_ptr.device != device:
            self.p_target_ptr = self.p_target_ptr.to(device)


def _normalize_p_target_type(value: str) -> str:
    normalized = str(value).strip()
    if normalized not in {"uniform", "model", "gt"}:
        raise ValueError("p_target_type must be one of uniform, model, gt.")
    return normalized


def _require_tensor_state(
    state: Mapping[str, Any],
    field_name: str,
    *,
    expected_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> Tensor:
    value = state.get(field_name)
    if not isinstance(value, Tensor):
        raise ValueError(f"Queue DA state requires tensor {field_name}.")
    tensor = value.detach().clone().to(device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Queue DA state {field_name} shape mismatch.")
    return tensor

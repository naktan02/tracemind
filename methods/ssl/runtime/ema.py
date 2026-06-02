"""Trainable-parameter EMA teacher runtime."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class EmaTeacherState:
    """EMA teacher checkpoint payload."""

    momentum: float
    parameter_names: tuple[str, ...]
    shadow: Mapping[str, Tensor]


class EmaTrainableParameterTeacher:
    """`requires_grad=True` model parameter에 대한 EMA teacher.

    USB EMA는 전체 model parameter/buffer shadow를 잡는다. TraceMind PEFT trainer는
    frozen backbone을 고정하므로 trainable parameter만 shadowing해도 teacher-student
    weight averaging 의미가 유지된다.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        momentum: float,
    ) -> None:
        if not 0.0 <= float(momentum) <= 1.0:
            raise ValueError("momentum must be in [0, 1].")
        self.momentum = float(momentum)
        self._parameter_names = self._trainable_parameter_names(model)
        if not self._parameter_names:
            raise ValueError("EMA teacher requires at least one trainable parameter.")
        self._shadow = {
            name: parameter.detach().clone()
            for name, parameter in self._named_trainable_parameters(model).items()
        }

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self._parameter_names

    def shadow_state(self) -> Mapping[str, Tensor]:
        """checkpoint에 넣을 detached CPU shadow tensor를 반환한다."""

        return {
            name: tensor.detach().cpu().clone() for name, tensor in self._shadow.items()
        }

    def state_dict(self) -> EmaTeacherState:
        """EMA teacher state를 checkpoint-safe value로 내보낸다."""

        return EmaTeacherState(
            momentum=self.momentum,
            parameter_names=self._parameter_names,
            shadow=self.shadow_state(),
        )

    def load_state_dict(
        self,
        *,
        shadow: Mapping[str, Tensor],
        parameter_names: tuple[str, ...] | None = None,
    ) -> None:
        """checkpoint shadow tensor를 현재 teacher에 복원한다."""

        expected_names = self._parameter_names
        actual_names = tuple(parameter_names or tuple(shadow.keys()))
        if set(actual_names) != set(expected_names):
            raise ValueError("EMA teacher state parameter names do not match model.")
        restored: dict[str, Tensor] = {}
        for name in expected_names:
            value = shadow.get(name)
            if not isinstance(value, Tensor):
                raise ValueError(f"EMA teacher state requires tensor for {name}.")
            restored[name] = value.detach().clone().to(self._shadow[name].device)
        self._shadow = restored

    def update(self, model: nn.Module) -> None:
        """student 최신 trainable parameter로 EMA shadow를 갱신한다."""

        parameters = self._named_trainable_parameters(model)
        self._require_matching_names(parameters)
        for name, parameter in parameters.items():
            self._shadow[name].mul_(self.momentum).add_(
                parameter.detach(),
                alpha=1.0 - self.momentum,
            )

    @contextmanager
    def use_shadow_weights(self, model: nn.Module) -> Iterator[None]:
        """teacher forward 동안 trainable parameter를 shadow 값으로 잠시 교체한다."""

        parameters = self._named_trainable_parameters(model)
        self._require_matching_names(parameters)
        backup = {
            name: parameter.detach().clone() for name, parameter in parameters.items()
        }
        try:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(self._shadow[name].to(parameter.device))
            yield
        finally:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(backup[name].to(parameter.device))

    @staticmethod
    def _named_trainable_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
        return {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @classmethod
    def _trainable_parameter_names(cls, model: nn.Module) -> tuple[str, ...]:
        return tuple(cls._named_trainable_parameters(model).keys())

    def _require_matching_names(
        self,
        parameters: Mapping[str, nn.Parameter],
    ) -> None:
        if set(parameters) != set(self._parameter_names):
            raise ValueError("EMA teacher model trainable parameter names changed.")

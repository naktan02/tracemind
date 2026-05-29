"""Shared adapter privacy guard contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from shared.src.contracts.training_contracts import TrainingTask
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)


@dataclass(slots=True)
class PrivacyProtectedUpdate:
    """privacy guard 적용 결과."""

    update: SharedAdapterUpdate
    clipped: bool = False
    dp_applied: bool = False


class SharedAdapterPrivacyGuard(Protocol):
    """Shared adapter update에 clipping/DP를 적용하는 method-owned interface."""

    guard_name: str
    supported_adapter_kinds: tuple[str, ...]

    def protect(
        self,
        *,
        update: SharedAdapterUpdate,
        training_task: TrainingTask,
    ) -> PrivacyProtectedUpdate:
        """update에 privacy/safety 보호 계층을 적용한다."""


PrivacyGuardFactory = Callable[[], SharedAdapterPrivacyGuard]

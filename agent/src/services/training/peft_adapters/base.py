"""PEFT adapter builder seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from torch import nn


@dataclass(frozen=True, slots=True)
class PeftAdapterBuildContext:
    """PEFT builder가 transformer/peft optional dependency를 쓰는 context."""

    cfg: Any
    lora_config_cls: type
    task_type: Any
    get_peft_model: Callable[[nn.Module, Any], nn.Module]


class PeftAdapterBuilder(Protocol):
    """Frozen backbone 위에 PEFT adapter를 얹는 strategy."""

    adapter_name: str

    def build_backbone(
        self,
        *,
        backbone_base: nn.Module,
        context: PeftAdapterBuildContext,
    ) -> nn.Module:
        """base transformer에 PEFT adapter를 적용한다."""

    def build_summary(self, *, cfg: Any) -> dict[str, Any]:
        """run manifest에 남길 PEFT adapter metadata를 만든다."""

"""LoRA-family PEFT adapter builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from methods.adaptation.peft_adapters.base import PeftAdapterBuildContext
from methods.adaptation.peft_adapters.registry import (
    peft_adapter_config_from_cfg,
    register_peft_adapter_builder,
)


def resolve_target_modules(raw_value: Any) -> str | list[str]:
    """Hydra scalar/list target_modules 값을 PEFT가 받는 형태로 정규화한다."""

    if isinstance(raw_value, str):
        return raw_value
    return [str(value) for value in raw_value]


@dataclass(frozen=True, slots=True)
class LoraPeftAdapterBuilder:
    """LoRA/RSLoRA 계열 PEFT adapter builder."""

    adapter_name: str
    use_rslora_override: bool | None = None

    def build_backbone(
        self,
        *,
        backbone_base: nn.Module,
        context: PeftAdapterBuildContext,
    ) -> nn.Module:
        peft_adapter = peft_adapter_config_from_cfg(context.cfg)
        lora_config = context.lora_config_cls(
            r=int(peft_adapter.rank),
            lora_alpha=int(peft_adapter.alpha),
            lora_dropout=float(peft_adapter.dropout),
            target_modules=resolve_target_modules(peft_adapter.target_modules),
            bias=str(peft_adapter.bias),
            use_rslora=self._use_rslora(cfg=context.cfg),
            task_type=context.task_type.FEATURE_EXTRACTION,
        )
        return context.get_peft_model(backbone_base, lora_config)

    def build_summary(self, *, cfg: Any) -> dict[str, Any]:
        peft_adapter = peft_adapter_config_from_cfg(cfg)
        return {
            "adapter_name": self.adapter_name,
            "rank": int(peft_adapter.rank),
            "alpha": int(peft_adapter.alpha),
            "dropout": float(peft_adapter.dropout),
            "bias": str(peft_adapter.bias),
            "target_modules": resolve_target_modules(peft_adapter.target_modules),
            "use_rslora": self._use_rslora(cfg=cfg),
        }

    def _use_rslora(self, *, cfg: Any) -> bool:
        if self.use_rslora_override is not None:
            return self.use_rslora_override
        return bool(peft_adapter_config_from_cfg(cfg).use_rslora)


@register_peft_adapter_builder("lora")
def build_lora_peft_adapter_builder() -> LoraPeftAdapterBuilder:
    """기본 LoRA builder를 생성한다."""

    return LoraPeftAdapterBuilder(adapter_name="lora")


@register_peft_adapter_builder("rslora")
def build_rslora_peft_adapter_builder() -> LoraPeftAdapterBuilder:
    """RSLoRA builder를 생성한다."""

    return LoraPeftAdapterBuilder(
        adapter_name="rslora",
        use_rslora_override=True,
    )

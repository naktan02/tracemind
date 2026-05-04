"""LoRA-family PEFT adapter builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from methods.adaptation.peft.base import PeftAdapterBuildContext
from methods.adaptation.peft.registry import register_peft_adapter_builder


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
        cfg = context.cfg
        lora_config = context.lora_config_cls(
            r=int(cfg.lora.rank),
            lora_alpha=int(cfg.lora.alpha),
            lora_dropout=float(cfg.lora.dropout),
            target_modules=resolve_target_modules(cfg.lora.target_modules),
            bias=str(cfg.lora.bias),
            use_rslora=self._use_rslora(cfg=cfg),
            task_type=context.task_type.FEATURE_EXTRACTION,
        )
        return context.get_peft_model(backbone_base, lora_config)

    def build_summary(self, *, cfg: Any) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "rank": int(cfg.lora.rank),
            "alpha": int(cfg.lora.alpha),
            "dropout": float(cfg.lora.dropout),
            "bias": str(cfg.lora.bias),
            "target_modules": resolve_target_modules(cfg.lora.target_modules),
            "use_rslora": self._use_rslora(cfg=cfg),
        }

    def _use_rslora(self, *, cfg: Any) -> bool:
        if self.use_rslora_override is not None:
            return self.use_rslora_override
        return bool(cfg.lora.use_rslora)


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

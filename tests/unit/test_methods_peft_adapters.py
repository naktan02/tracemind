"""Query adaptation PEFT adapter builder tests."""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from methods.adaptation.peft_adapters.base import (
    PeftAdapterBuildContext,
)
from methods.adaptation.peft_adapters.registry import (
    build_peft_adapter_builder,
    resolve_peft_adapter_name,
)


def _cfg(**overrides):
    values = {
        "peft_adapter_name": "lora",
        "rank": 8,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": "all-linear",
        "bias": "none",
        "use_rslora": False,
    }
    values.update(overrides)
    return SimpleNamespace(lora=SimpleNamespace(**values))


def test_peft_adapter_registry_builds_lora_summary() -> None:
    cfg = _cfg()

    builder = build_peft_adapter_builder(resolve_peft_adapter_name(cfg))

    assert builder.adapter_name == "lora"
    assert builder.build_summary(cfg=cfg) == {
        "adapter_name": "lora",
        "rank": 8,
        "alpha": 16,
        "dropout": 0.1,
        "bias": "none",
        "target_modules": "all-linear",
        "use_rslora": False,
    }


def test_peft_adapter_registry_builds_rslora_with_forced_flag() -> None:
    cfg = _cfg(peft_adapter_name="rslora", use_rslora=False)

    builder = build_peft_adapter_builder(resolve_peft_adapter_name(cfg))

    assert builder.adapter_name == "rslora"
    assert builder.build_summary(cfg=cfg)["use_rslora"] is True


def test_lora_builder_delegates_to_peft_stack() -> None:
    captured = {}

    class _TaskType:
        FEATURE_EXTRACTION = "feature-extraction"

    class _LoraConfig:
        def __init__(self, **kwargs) -> None:
            captured["lora_config"] = kwargs

    def _get_peft_model(backbone, peft_config):
        captured["backbone"] = backbone
        captured["peft_config"] = peft_config
        return "peft-backbone"

    cfg = _cfg(target_modules=["query", "value"])
    backbone_base = nn.Linear(2, 2)
    builder = build_peft_adapter_builder("lora")

    backbone = builder.build_backbone(
        backbone_base=backbone_base,
        context=PeftAdapterBuildContext(
            cfg=cfg,
            lora_config_cls=_LoraConfig,
            task_type=_TaskType,
            get_peft_model=_get_peft_model,
        ),
    )

    assert backbone == "peft-backbone"
    assert captured["backbone"] is backbone_base
    assert captured["lora_config"]["target_modules"] == ["query", "value"]
    assert captured["lora_config"]["task_type"] == "feature-extraction"

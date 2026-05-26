"""Compatibility shim for legacy LoRA PEFT adapter builder imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_adapters.lora.builder import (
    LoraPeftAdapterBuilder,
    build_lora_peft_adapter_builder,
    build_rslora_peft_adapter_builder,
    resolve_target_modules,
)

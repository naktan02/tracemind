"""Compatibility shim for legacy PEFT adapter registry imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_adapters.registry import (
    PeftAdapterBuilderFactory,
    build_peft_adapter_builder,
    register_peft_adapter_builder,
    resolve_peft_adapter_name,
)

"""PEFT adapter builder registry."""

from __future__ import annotations

from collections.abc import Callable

from .base import PeftAdapterBuilder

PeftAdapterBuilderFactory = Callable[[], PeftAdapterBuilder]

_PEFT_ADAPTER_BUILDER_REGISTRY: dict[str, PeftAdapterBuilderFactory] = {}


def register_peft_adapter_builder(
    *adapter_names: str,
) -> Callable[[PeftAdapterBuilderFactory], PeftAdapterBuilderFactory]:
    """PEFT adapter builder factory를 이름으로 등록한다."""

    def _decorator(
        factory: PeftAdapterBuilderFactory,
    ) -> PeftAdapterBuilderFactory:
        for adapter_name in adapter_names:
            _PEFT_ADAPTER_BUILDER_REGISTRY[adapter_name.strip().lower()] = factory
        return factory

    return _decorator


def resolve_peft_adapter_name(cfg) -> str:
    """Hydra cfg에서 PEFT adapter 이름을 해석한다."""

    explicit_name = str(
        getattr(cfg.lora, "peft_adapter_name", "")
        or getattr(cfg.lora, "adapter_name", "")
        or getattr(cfg.lora, "method", "")
        or ""
    ).strip()
    if explicit_name:
        return explicit_name
    if bool(getattr(cfg.lora, "use_rslora", False)):
        return "rslora"
    return "lora"


def build_peft_adapter_builder(adapter_name: str) -> PeftAdapterBuilder:
    """PEFT adapter 이름으로 builder를 생성한다."""

    normalized_name = adapter_name.strip().lower()
    factory = _PEFT_ADAPTER_BUILDER_REGISTRY.get(normalized_name)
    if factory is None:
        raise ValueError(f"Unsupported PEFT adapter builder: {adapter_name}.")
    return factory()


# Built-in PEFT adapters self-register via decorators when imported.
from . import lora as _lora  # noqa: E402,F401

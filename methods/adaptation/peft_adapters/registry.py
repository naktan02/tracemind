"""PEFT adapter builder registry."""

from __future__ import annotations

import pkgutil
from collections.abc import Callable
from importlib import import_module

from methods.adaptation.peft_adapters.base import PeftAdapterBuilder

PeftAdapterBuilderFactory = Callable[[], PeftAdapterBuilder]

_PEFT_ADAPTERS_PACKAGE = "methods.adaptation.peft_adapters"
_PEFT_ADAPTER_BUILDER_REGISTRY: dict[str, PeftAdapterBuilderFactory] = {}
_PEFT_ADAPTER_BUILDERS_LOADED = False


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


def peft_adapter_config_from_cfg(cfg):
    """Hydra/root config에서 PEFT adapter parameter surface를 찾는다."""

    if hasattr(cfg, "peft_adapter"):
        return cfg.peft_adapter
    raise ValueError("PEFT adapter config must define peft_adapter.")


def resolve_peft_adapter_name(cfg) -> str:
    """Hydra cfg에서 PEFT adapter 이름을 해석한다."""

    peft_adapter = peft_adapter_config_from_cfg(cfg)
    explicit_name = str(
        getattr(peft_adapter, "peft_adapter_name", "")
        or getattr(peft_adapter, "adapter_name", "")
        or getattr(peft_adapter, "method", "")
        or ""
    ).strip()
    if explicit_name:
        return explicit_name
    raise ValueError("PEFT adapter config must define peft_adapter_name.")


def build_peft_adapter_builder(adapter_name: str) -> PeftAdapterBuilder:
    """PEFT adapter 이름으로 builder를 생성한다."""

    load_builtin_peft_adapter_builders()
    normalized_name = adapter_name.strip().lower()
    factory = _PEFT_ADAPTER_BUILDER_REGISTRY.get(normalized_name)
    if factory is None:
        raise ValueError(f"Unsupported PEFT adapter builder: {adapter_name}.")
    return factory()


def load_builtin_peft_adapter_builders() -> None:
    """built-in PEFT adapter builder module을 convention으로 import한다."""

    global _PEFT_ADAPTER_BUILDERS_LOADED
    if _PEFT_ADAPTER_BUILDERS_LOADED:
        return

    package = import_module(_PEFT_ADAPTERS_PACKAGE)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        _PEFT_ADAPTER_BUILDERS_LOADED = True
        return

    for module_info in pkgutil.iter_modules(package_paths):
        if not module_info.ispkg:
            continue
        _try_import_peft_adapter_builder_module(module_info.name)
    _PEFT_ADAPTER_BUILDERS_LOADED = True


def _try_import_peft_adapter_builder_module(package_name: str) -> bool:
    adapter_package = f"{_PEFT_ADAPTERS_PACKAGE}.{package_name}"
    module_name = f"{adapter_package}.builder"
    try:
        import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name not in {adapter_package, module_name}:
            raise
        return False
    return True

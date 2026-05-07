"""Round family registry primitive."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping

from ..aggregation.diagonal_scale_defaults import AggregationConfigScalar
from .models import RoundFamilyFactory, SharedAdapterRoundFamily

_ROUND_FAMILY_REGISTRY: dict[str, RoundFamilyFactory] = {}


def register_shared_adapter_round_family(
    *family_names: str,
    factory: RoundFamilyFactory | None = None,
) -> (
    Callable[[RoundFamilyFactory], RoundFamilyFactory]
    | RoundFamilyFactory
):
    """adapter family factory 옆에서 round runtime wiring을 등록한다."""

    def _decorator(factory: RoundFamilyFactory) -> RoundFamilyFactory:
        for family_name in family_names:
            _ROUND_FAMILY_REGISTRY[family_name.strip().lower()] = factory
        return factory

    if factory is not None:
        return _decorator(factory)
    return _decorator


def build_shared_adapter_round_family(
    family_name: str,
    *,
    aggregation_backend_name: str,
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] | None = None,
) -> SharedAdapterRoundFamily:
    """adapter family와 aggregation backend 이름으로 서버 조합 객체를 만든다."""

    normalized_family_name = family_name.strip().lower()
    _import_round_family_module_by_convention(normalized_family_name)
    factory = _ROUND_FAMILY_REGISTRY.get(normalized_family_name)
    if factory is not None:
        return factory(aggregation_backend_name, aggregation_backend_overrides)
    raise ValueError(f"Unsupported shared adapter family: {family_name}.")


def _import_round_family_module_by_convention(normalized_family_name: str) -> None:
    """family name과 module name이 같은 builtin family를 필요할 때 import한다."""

    module_name = normalized_family_name.replace("-", "_")
    try:
        importlib.import_module(
            f"main_server.src.services.federation.rounds.families.{module_name}"
        )
    except ModuleNotFoundError as error:
        expected_module = (
            "main_server.src.services.federation.rounds.families."
            f"{module_name}"
        )
        if error.name != expected_module:
            raise

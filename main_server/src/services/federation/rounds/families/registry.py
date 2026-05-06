"""Round family registry."""

from __future__ import annotations

from collections.abc import Mapping

from shared.src.config.adapter_family_metadata import (
    CLASSIFIER_HEAD_FAMILY_METADATA,
    DIAGONAL_SCALE_FAMILY_METADATA,
    LORA_CLASSIFIER_FAMILY_METADATA,
)

from ..aggregation.diagonal_scale_defaults import AggregationConfigScalar
from ..aggregation.registry import (
    build_shared_adapter_aggregation_backend,
)
from .classifier_head import ClassifierHeadRoundFamily
from .diagonal_scale import DiagonalScaleRoundFamily
from .lora_classifier import LoraClassifierRoundFamily
from .models import RoundFamilyFactory, SharedAdapterRoundFamily

_ROUND_FAMILY_REGISTRY: dict[str, RoundFamilyFactory] = {}


def register_shared_adapter_round_family(
    *family_names: str,
    factory: RoundFamilyFactory,
) -> None:
    """adapter family 조합 factory를 얇은 wiring registry에 등록한다."""

    for family_name in family_names:
        _ROUND_FAMILY_REGISTRY[family_name.strip().lower()] = factory


def build_shared_adapter_round_family(
    family_name: str,
    *,
    aggregation_backend_name: str,
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] | None = None,
) -> SharedAdapterRoundFamily:
    """adapter family와 aggregation backend 이름으로 서버 조합 객체를 만든다."""

    normalized_family_name = family_name.strip().lower()
    factory = _ROUND_FAMILY_REGISTRY.get(normalized_family_name)
    if factory is not None:
        return factory(aggregation_backend_name, aggregation_backend_overrides)
    raise ValueError(f"Unsupported shared adapter family: {family_name}.")


def _build_diagonal_scale_round_family(
    aggregation_backend_name: str,
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] | None,
) -> SharedAdapterRoundFamily:
    return DiagonalScaleRoundFamily(
        aggregation_backend=build_shared_adapter_aggregation_backend(
            adapter_kind=DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,
            backend_name=aggregation_backend_name,
            overrides=aggregation_backend_overrides,
        )
    )


def _build_classifier_head_round_family(
    aggregation_backend_name: str,
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] | None,
) -> SharedAdapterRoundFamily:
    return ClassifierHeadRoundFamily(
        aggregation_backend=build_shared_adapter_aggregation_backend(
            adapter_kind=CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,
            backend_name=aggregation_backend_name,
            overrides=aggregation_backend_overrides,
        )
    )


def _build_lora_classifier_round_family(
    aggregation_backend_name: str,
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] | None,
) -> SharedAdapterRoundFamily:
    return LoraClassifierRoundFamily(
        aggregation_backend=build_shared_adapter_aggregation_backend(
            adapter_kind=LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind,
            backend_name=aggregation_backend_name,
            overrides=aggregation_backend_overrides,
        )
    )


register_shared_adapter_round_family(
    DIAGONAL_SCALE_FAMILY_METADATA.family_name,
    factory=_build_diagonal_scale_round_family,
)
register_shared_adapter_round_family(
    CLASSIFIER_HEAD_FAMILY_METADATA.family_name,
    factory=_build_classifier_head_round_family,
)
register_shared_adapter_round_family(
    LORA_CLASSIFIER_FAMILY_METADATA.family_name,
    factory=_build_lora_classifier_round_family,
)

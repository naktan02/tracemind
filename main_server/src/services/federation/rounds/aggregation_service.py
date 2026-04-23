"""Shared adapter 집계 서비스.

현재 concrete 구현은 diagonal scale adapter 하나만 제공한다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Sequence

from main_server.src.services.federation.rounds.diagonal_scale_defaults import (
    DEFAULT_DIAGONAL_SCALE_FEDAVG_AGGREGATION_CONFIG,
    AggregationConfigScalar,
    DiagonalScaleFedAvgAggregationConfig,
)
from shared.src.config.adapter_family_metadata import (
    CLASSIFIER_HEAD_FAMILY_METADATA,
    DIAGONAL_SCALE_FAMILY_METADATA,
)
from shared.src.config.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)
from shared.src.contracts.adapter_contracts import (
    ClassifierHeadDelta,
    ClassifierHeadState,
    VectorAdapterDelta,
    VectorAdapterState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

AggregationConfig = DiagonalScaleFedAvgAggregationConfig


@dataclass(slots=True)
class AggregationResult:
    """집계 결과로 만들어진 새 전역 상태와 메트릭."""

    next_state: SharedAdapterState
    aggregated_metrics: dict[str, float]
    update_count: int


class SharedAdapterAggregationBackend(Protocol):
    """Shared adapter 종류별 서버 집계 backend 인터페이스."""

    adapter_kind: str

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ) -> AggregationResult:
        """같은 adapter family의 update들을 새 전역 상태로 합친다."""


AggregationBackendFactory = Callable[
    [Mapping[str, AggregationConfigScalar] | None],
    SharedAdapterAggregationBackend,
]


@dataclass(slots=True)
class DiagonalScaleAggregationService:
    """Diagonal scale adapter update를 전역 상태로 집계한다."""

    adapter_kind: str = DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind
    config: AggregationConfig = field(
        default_factory=lambda: DEFAULT_DIAGONAL_SCALE_FEDAVG_AGGREGATION_CONFIG
    )

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, AggregationConfigScalar] | None,
    ) -> "DiagonalScaleAggregationService":
        return cls(config=AggregationConfig.from_mapping(source))

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ) -> AggregationResult:
        if not isinstance(base_state, VectorAdapterState):
            raise TypeError(
                "DiagonalScaleAggregationService expects VectorAdapterState as the "
                f"base state, got {type(base_state)!r}."
            )
        if base_state.adapter_kind != self.adapter_kind:
            raise ValueError(
                "Base state adapter_kind does not match the diagonal scale "
                f"aggregator: {base_state.adapter_kind}"
            )

        valid_updates = [
            payload for payload in update_payloads if payload.example_count > 0
        ]
        if not valid_updates:
            raise ValueError("At least one non-empty update payload is required.")

        embedding_dim = base_state.embedding_dim
        total_examples = sum(payload.example_count for payload in valid_updates)
        weighted_delta = [0.0] * embedding_dim
        weighted_confidence = 0.0
        weighted_margin = 0.0
        weighted_delta_norm = 0.0

        for payload in valid_updates:
            if not isinstance(payload, VectorAdapterDelta):
                raise TypeError(
                    "DiagonalScaleAggregationService expects VectorAdapterDelta "
                    f"updates, got {type(payload)!r}."
                )
            if payload.adapter_kind != self.adapter_kind:
                raise ValueError(
                    "Update adapter_kind does not match the diagonal scale "
                    f"aggregator: {payload.adapter_kind}"
                )
            if payload.model_id != base_state.model_id:
                raise ValueError("All update payloads must match the base model_id.")
            if payload.base_model_revision != base_state.model_revision:
                raise ValueError(
                    "All update payloads must match the base model revision."
                )
            if payload.training_scope != base_state.training_scope:
                raise ValueError("All update payloads must match the training scope.")
            if payload.embedding_dim != embedding_dim:
                raise ValueError(
                    "All update payloads must share the same embedding_dim."
                )

            weight = payload.example_count / total_examples
            weighted_confidence += payload.mean_confidence * weight
            weighted_margin += (payload.mean_margin or 0.0) * weight
            weighted_delta_norm += payload.l2_norm() * weight
            for index, value in enumerate(payload.dimension_deltas):
                weighted_delta[index] += value * weight

        next_scales = [
            max(
                self.config.min_scale,
                min(self.config.max_scale, scale + delta),
            )
            for scale, delta in zip(
                base_state.dimension_scales,
                weighted_delta,
                strict=True,
            )
        ]
        next_state = VectorAdapterState(
            schema_version=base_state.schema_version,
            model_id=base_state.model_id,
            model_revision=next_model_revision,
            training_scope=base_state.training_scope,
            dimension_scales=next_scales,
            updated_at=aggregated_at,
            adapter_kind=base_state.adapter_kind,
        )
        return AggregationResult(
            next_state=next_state,
            aggregated_metrics={
                "client_count": float(len(valid_updates)),
                "example_count": float(total_examples),
                "mean_confidence": weighted_confidence,
                "mean_margin": weighted_margin,
                "mean_delta_l2_norm": weighted_delta_norm,
            },
            update_count=len(valid_updates),
        )


@dataclass(slots=True)
class ClassifierHeadFedAvgAggregationService:
    """Classifier-head delta를 전역 선형 head 상태로 집계한다."""

    adapter_kind: str = CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ) -> AggregationResult:
        if not isinstance(base_state, ClassifierHeadState):
            raise TypeError(
                "ClassifierHeadFedAvgAggregationService expects "
                f"ClassifierHeadState as the base state, got {type(base_state)!r}."
            )
        if base_state.adapter_kind != self.adapter_kind:
            raise ValueError(
                "Base state adapter_kind does not match the classifier-head "
                f"aggregator: {base_state.adapter_kind}"
            )

        valid_updates = [
            payload for payload in update_payloads if payload.example_count > 0
        ]
        if not valid_updates:
            raise ValueError("At least one non-empty update payload is required.")

        labels = base_state.labels
        embedding_dim = base_state.embedding_dim
        total_examples = sum(payload.example_count for payload in valid_updates)
        weighted_weight_deltas = {
            label: [0.0] * embedding_dim for label in labels
        }
        weighted_bias_deltas = {label: 0.0 for label in labels}
        weighted_confidence = 0.0
        weighted_margin = 0.0
        weighted_delta_norm = 0.0

        for payload in valid_updates:
            if not isinstance(payload, ClassifierHeadDelta):
                raise TypeError(
                    "ClassifierHeadFedAvgAggregationService expects "
                    f"ClassifierHeadDelta updates, got {type(payload)!r}."
                )
            if payload.adapter_kind != self.adapter_kind:
                raise ValueError(
                    "Update adapter_kind does not match the classifier-head "
                    f"aggregator: {payload.adapter_kind}"
                )
            if payload.model_id != base_state.model_id:
                raise ValueError("All update payloads must match the base model_id.")
            if payload.base_model_revision != base_state.model_revision:
                raise ValueError(
                    "All update payloads must match the base model revision."
                )
            if payload.training_scope != base_state.training_scope:
                raise ValueError("All update payloads must match the training scope.")
            if payload.labels != labels:
                raise ValueError(
                    "Classifier head updates must share the same ordered labels."
                )
            if payload.embedding_dim != embedding_dim:
                raise ValueError(
                    "All update payloads must share the same embedding_dim."
                )

            weight = payload.example_count / total_examples
            weighted_confidence += payload.mean_confidence * weight
            weighted_margin += (payload.mean_margin or 0.0) * weight
            weighted_delta_norm += payload.l2_norm() * weight
            for label in labels:
                for index, value in enumerate(payload.label_weight_deltas[label]):
                    weighted_weight_deltas[label][index] += float(value) * weight
                weighted_bias_deltas[label] += (
                    float(payload.label_bias_deltas.get(label, 0.0)) * weight
                )

        next_state = ClassifierHeadState(
            schema_version=base_state.schema_version,
            adapter_kind=base_state.adapter_kind,
            model_id=base_state.model_id,
            model_revision=next_model_revision,
            training_scope=base_state.training_scope,
            updated_at=aggregated_at,
            label_weights={
                label: [
                    float(base_value) + float(delta)
                    for base_value, delta in zip(
                        base_state.label_weights[label],
                        weighted_weight_deltas[label],
                        strict=True,
                    )
                ]
                for label in labels
            },
            label_biases={
                label: float(base_state.label_biases.get(label, 0.0))
                + float(weighted_bias_deltas[label])
                for label in labels
            },
        )
        return AggregationResult(
            next_state=next_state,
            aggregated_metrics={
                "client_count": float(len(valid_updates)),
                "example_count": float(total_examples),
                "mean_confidence": weighted_confidence,
                "mean_margin": weighted_margin,
                "mean_delta_l2_norm": weighted_delta_norm,
            },
            update_count=len(valid_updates),
        )


AggregationService = DiagonalScaleAggregationService

_AGGREGATION_BACKEND_REGISTRY: dict[
    tuple[str, str],
    tuple[AggregationBackendFactory, RegistryCatalogEntry],
] = {}


def register_shared_adapter_aggregation_backend(
    adapter_kind: str,
    *backend_names: str,
    factory: AggregationBackendFactory,
    catalog_entry: RegistryCatalogEntry,
) -> None:
    """adapter family별 aggregation backend를 얇은 wiring registry에 등록한다."""
    normalized_adapter_kind = adapter_kind.strip().lower()
    registered_backend = (factory, catalog_entry)
    for backend_name in backend_names:
        normalized_backend_name = backend_name.strip().lower()
        _AGGREGATION_BACKEND_REGISTRY[
            (normalized_adapter_kind, normalized_backend_name)
        ] = registered_backend


def build_shared_adapter_aggregation_backend(
    *,
    adapter_kind: str,
    backend_name: str,
    overrides: Mapping[str, AggregationConfigScalar] | None = None,
) -> SharedAdapterAggregationBackend:
    """adapter family와 backend 이름으로 aggregation backend를 조립한다."""

    normalized_key = (adapter_kind.strip().lower(), backend_name.strip().lower())
    registered_backend = _AGGREGATION_BACKEND_REGISTRY.get(normalized_key)
    if registered_backend is not None:
        factory, _catalog_entry = registered_backend
        return factory(overrides)
    raise ValueError(
        "Unsupported aggregation backend for adapter family: "
        f"adapter_kind={adapter_kind}, backend_name={backend_name}"
    )


def list_registered_shared_adapter_aggregation_backends(
    *,
    adapter_kind: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """등록된 aggregation backend 키를 정렬된 tuple로 반환한다."""

    normalized_adapter_kind = None
    if adapter_kind is not None:
        normalized_adapter_kind = adapter_kind.strip().lower()
    registered = sorted(_AGGREGATION_BACKEND_REGISTRY)
    if normalized_adapter_kind is None:
        return tuple(registered)
    return tuple(
        key
        for key in registered
        if key[0] == normalized_adapter_kind
    )


def list_shared_adapter_aggregation_backend_catalog_entries(
) -> tuple[RegistryCatalogEntry, ...]:
    """등록된 aggregation backend catalog entry를 canonical item 기준으로 반환한다."""

    return dedupe_registry_catalog_entries(
        catalog_entry
        for _factory, catalog_entry in _AGGREGATION_BACKEND_REGISTRY.values()
    )


register_shared_adapter_aggregation_backend(
    DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,
    "fedavg",
    "diagonal_scale_fedavg",
    factory=DiagonalScaleAggregationService.from_mapping,
    catalog_entry=RegistryCatalogEntry(
        item_name=f"{DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind}.fedavg",
        display_name="fedavg",
        implementation_module=DiagonalScaleAggregationService.__module__,
        core_method_name="fedavg",
        family_name=DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,
        supported_adapter_kinds=(DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,),
        metadata={"adapter_kind": DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind},
    ),
)
register_shared_adapter_aggregation_backend(
    CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,
    "fedavg",
    "classifier_head_fedavg",
    factory=lambda overrides: ClassifierHeadFedAvgAggregationService(),
    catalog_entry=RegistryCatalogEntry(
        item_name=f"{CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind}.fedavg",
        display_name="fedavg",
        implementation_module=ClassifierHeadFedAvgAggregationService.__module__,
        core_method_name="fedavg",
        family_name=CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,
        supported_adapter_kinds=(CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,),
        metadata={"adapter_kind": CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind},
    ),
)


__all__ = [
    "AggregationBackendFactory",
    "AggregationConfig",
    "AggregationResult",
    "AggregationService",
    "ClassifierHeadFedAvgAggregationService",
    "DiagonalScaleAggregationService",
    "SharedAdapterAggregationBackend",
    "build_shared_adapter_aggregation_backend",
    "list_shared_adapter_aggregation_backend_catalog_entries",
    "list_registered_shared_adapter_aggregation_backends",
    "register_shared_adapter_aggregation_backend",
]

"""Federated aggregation method metadata and strategy contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, cast

from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

AggregationConfigScalar = str | int | float | bool
AGGREGATED_ARTIFACT_FORMAT_KEY = "artifact_format"
AGGREGATED_ARTIFACT_TENSORS_KEY = "tensors"
AGGREGATED_ARTIFACT_METADATA_KEY = "metadata"
AGGREGATED_SAFETENSORS_ARTIFACT_FORMAT = "safetensors"


def build_safetensors_aggregated_artifact(
    *,
    tensors: Mapping[str, object],
    metadata: Mapping[str, str],
) -> dict[str, object]:
    """server-owned aggregate tensor artifact 저장 envelope을 만든다."""

    return {
        AGGREGATED_ARTIFACT_FORMAT_KEY: AGGREGATED_SAFETENSORS_ARTIFACT_FORMAT,
        AGGREGATED_ARTIFACT_TENSORS_KEY: dict(tensors),
        AGGREGATED_ARTIFACT_METADATA_KEY: dict(metadata),
    }


def is_safetensors_aggregated_artifact(payload: Mapping[str, object]) -> bool:
    """aggregate artifact payload가 safetensors envelope인지 확인한다."""

    return (
        payload.get(AGGREGATED_ARTIFACT_FORMAT_KEY)
        == AGGREGATED_SAFETENSORS_ARTIFACT_FORMAT
    )


def safetensors_aggregated_artifact_parts(
    payload: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, str]]:
    """safetensors aggregate envelope에서 tensor와 metadata를 꺼낸다."""

    tensors = payload.get(AGGREGATED_ARTIFACT_TENSORS_KEY)
    metadata = payload.get(AGGREGATED_ARTIFACT_METADATA_KEY)
    if not isinstance(tensors, Mapping):
        raise ValueError("safetensors aggregate artifact requires tensors mapping.")
    if not isinstance(metadata, Mapping):
        raise ValueError("safetensors aggregate artifact requires metadata mapping.")
    return tensors, cast(Mapping[str, str], metadata)


@dataclass(frozen=True, slots=True)
class FederatedAggregationMethodSpec:
    """server lifecycle과 분리된 aggregation method catalog 항목."""

    adapter_kind: str
    method_name: str
    implementation_module: str
    core_function_name: str
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, AggregationConfigScalar | None] | None = None


class AggregatedArtifactRefResolver(Protocol):
    """server-owned aggregate artifact ref 생성 capability."""

    artifact_format: str

    def build_ref(
        self,
        *,
        next_model_revision: str,
        artifact_name: str,
    ) -> str:
        """logical artifact name을 server-owned opaque ref로 변환한다."""


class AggregationJsonArtifactLoader(Protocol):
    """server-owned artifact ref에서 JSON mapping artifact를 읽는 capability."""

    def load_json_artifact(
        self,
        *,
        artifact_ref: str,
    ) -> Mapping[str, object]:
        """opaque artifact ref를 JSON mapping으로 materialize한다."""


@dataclass(frozen=True, slots=True)
class FederatedAggregationContext:
    """server runtime이 strategy에 제공하는 실행 context."""

    next_model_revision: str
    aggregated_at: datetime
    artifact_ref_resolver: AggregatedArtifactRefResolver | None = None
    artifact_loader: AggregationJsonArtifactLoader | None = None

    def require_artifact_ref_resolver(
        self,
        *,
        context: str,
    ) -> AggregatedArtifactRefResolver:
        """artifact materialization capability가 필요한 strategy에서 호출한다."""

        if self.artifact_ref_resolver is None:
            raise ValueError(f"{context} requires an aggregate artifact ref resolver.")
        return self.artifact_ref_resolver

    def require_artifact_loader(
        self,
        *,
        context: str,
    ) -> AggregationJsonArtifactLoader:
        """artifact-ref update materialization이 필요한 strategy에서 호출한다."""

        if self.artifact_loader is None:
            raise ValueError(f"{context} requires an artifact materializer.")
        return self.artifact_loader


@dataclass(frozen=True, slots=True)
class FederatedAggregationResult:
    """methods-owned aggregation strategy 실행 결과."""

    next_state: SharedAdapterState
    aggregated_metrics: dict[str, float]
    update_count: int
    aggregated_artifacts: dict[str, dict[str, object]] = field(default_factory=dict)


class FederatedAggregationStrategy(Protocol):
    """payload adapter별 update들을 다음 shared state로 집계하는 strategy."""

    adapter_kind: str
    method_name: str

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        context: FederatedAggregationContext,
    ) -> FederatedAggregationResult:
        """같은 payload adapter update들을 다음 state로 집계한다."""


FederatedAggregationStrategyFactory = Callable[
    [Mapping[str, AggregationConfigScalar] | None],
    FederatedAggregationStrategy,
]

"""Methods-owned aggregation strategy executor for main_server runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from methods.federated.aggregation.base import (
    FederatedAggregationContext,
    FederatedAggregationStrategy,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from .artifact_refs import AggregatedArtifactRefBuilder, AggregationArtifactStore
from .models import AggregationConfigScalar, AggregationResult

DEFAULT_AGGREGATED_ARTIFACT_FORMAT = "server_aggregated_artifact_ref"


@dataclass(slots=True)
class MethodAggregationBackend:
    """main_server boundary를 methods-owned aggregation strategy에 연결한다."""

    strategy: FederatedAggregationStrategy
    overrides: Mapping[str, AggregationConfigScalar] | None = None
    artifact_loader: AggregationArtifactStore = field(
        default_factory=AggregationArtifactStore
    )
    artifact_ref_builder: AggregatedArtifactRefBuilder = field(init=False)

    def __post_init__(self) -> None:
        self.artifact_ref_builder = _build_aggregated_artifact_ref_builder(
            adapter_kind=self.strategy.adapter_kind,
            source=self.overrides,
        )

    @property
    def adapter_kind(self) -> str:
        """이 backend가 처리하는 shared adapter family discriminator."""

        return self.strategy.adapter_kind

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ) -> AggregationResult:
        """server lifecycle context를 method strategy context로 변환한다."""

        method_result = self.strategy.aggregate(
            base_state=base_state,
            update_payloads=update_payloads,
            context=FederatedAggregationContext(
                next_model_revision=next_model_revision,
                aggregated_at=aggregated_at,
                artifact_ref_resolver=self.artifact_ref_builder,
                artifact_loader=self.artifact_loader,
            ),
        )
        self._save_aggregated_artifacts(method_result.aggregated_artifacts)
        return AggregationResult(
            next_state=method_result.next_state,
            aggregated_metrics=method_result.aggregated_metrics,
            update_count=method_result.update_count,
            aggregated_artifacts=method_result.aggregated_artifacts,
        )

    def _save_aggregated_artifacts(
        self,
        artifacts: Mapping[str, Mapping[str, object]],
    ) -> None:
        """methods strategy가 만든 server-owned aggregate artifact를 저장한다."""

        for artifact_ref, payload in artifacts.items():
            self.artifact_loader.save_json_artifact_ref(
                artifact_ref=artifact_ref,
                payload=dict(payload),
            )


def _build_aggregated_artifact_ref_builder(
    *,
    adapter_kind: str,
    source: Mapping[str, AggregationConfigScalar] | None,
) -> AggregatedArtifactRefBuilder:
    default_prefix = f"server-aggregate://{adapter_kind}"
    return AggregatedArtifactRefBuilder(
        artifact_ref_prefix=str(
            (source or {}).get("artifact_ref_prefix", default_prefix)
        ).strip(),
        artifact_format=str(
            (source or {}).get(
                "artifact_format",
                DEFAULT_AGGREGATED_ARTIFACT_FORMAT,
            )
        ).strip(),
    )

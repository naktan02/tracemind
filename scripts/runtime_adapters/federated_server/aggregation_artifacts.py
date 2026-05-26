"""Simulation server aggregation artifact bridge."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregatedArtifactRefBuilder,
    AggregationArtifactStore,
)
from main_server.src.services.federation.rounds.aggregation.executor import (
    DEFAULT_AGGREGATED_ARTIFACT_FORMAT,
)
from methods.federated.aggregation.base import FederatedAggregationContext


@dataclass(frozen=True, slots=True)
class ServerAggregationArtifactRefs:
    """server-owned aggregation artifact ref 묶음."""

    refs_by_name: dict[str, str]
    artifact_format: str


def build_server_aggregate_artifact_refs(
    *,
    adapter_family_name: str,
    next_model_revision: str,
    artifact_names: Sequence[str],
    artifact_format: str = DEFAULT_AGGREGATED_ARTIFACT_FORMAT,
) -> ServerAggregationArtifactRefs:
    """adapter family state publication에 필요한 server-owned artifact refs를 만든다."""

    artifact_ref_builder = AggregatedArtifactRefBuilder(
        artifact_ref_prefix=f"server-aggregate://{adapter_family_name}",
        artifact_format=artifact_format,
    )
    return ServerAggregationArtifactRefs(
        refs_by_name={
            artifact_name: artifact_ref_builder.build_ref(
                next_model_revision=next_model_revision,
                artifact_name=artifact_name,
            )
            for artifact_name in artifact_names
        },
        artifact_format=artifact_ref_builder.artifact_format,
    )


def build_simulation_aggregation_context(
    *,
    output_dir: Path,
    next_model_revision: str,
    aggregated_at: datetime,
) -> FederatedAggregationContext:
    """simulation output dir의 server-owned aggregation artifact store를 연결한다."""

    return FederatedAggregationContext(
        next_model_revision=next_model_revision,
        aggregated_at=aggregated_at,
        artifact_loader=AggregationArtifactStore(
            state_root=output_dir / "main_server" / "aggregation_artifacts"
        ),
    )

"""FL simulation 내부 실행 context 모델."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from methods.common.runtime_resources import RuntimeResourceCache
from methods.federated_ssl.hooks.peer_context import FederatedSslPeerClientSnapshot
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    FederatedClientShard,
    FederatedDatasetSplit,
    FederatedPeerProbeManifest,
    SimulationEvaluation,
    SimulationRoundSummary,
)
from scripts.experiments.fl_ssl.federated_simulation.runtime_resources import (
    RoundBaseSnapshotCache,
)
from scripts.runtime_adapters.federated_server.runtime import SimulationServerRuntime
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState


@dataclass(frozen=True, slots=True)
class ActiveSimulationState:
    """현재 round에서 client가 읽는 active global 상태."""

    manifest: ModelManifest
    adapter_state: SharedAdapterState


@dataclass(frozen=True, slots=True)
class PeerContextSimulationState:
    """simulation round 사이에 유지하는 peer selection/prediction snapshot."""

    client_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] = field(
        default_factory=dict
    )

    def selection_vectors(self) -> dict[str, tuple[float, ...]]:
        return {
            client_id: snapshot.selection_vector
            for client_id, snapshot in self.client_snapshots.items()
        }


@dataclass(frozen=True, slots=True)
class ClientPartitionSyncSimulationState:
    """round 사이에 유지하는 client-local partitioned trainable snapshot."""

    client_partition_snapshots: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )

    def snapshot_for_client(
        self,
        client_id: str,
    ) -> Mapping[str, Any]:
        return self.client_partition_snapshots.get(client_id, {})


@dataclass(frozen=True, slots=True)
class QuerySslAlgorithmSyncSimulationState:
    """round 사이에 유지하는 client-local Query SSL algorithm state."""

    client_algorithm_states: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )

    def state_for_client(self, client_id: str) -> Mapping[str, Any]:
        return self.client_algorithm_states.get(client_id, {})


@dataclass(frozen=True, slots=True)
class BootstrappedSimulation:
    """bootstrap 이후 FL simulation loop에 필요한 context."""

    dataset_split: FederatedDatasetSplit
    validation_client_shards: tuple[FederatedClientShard, ...]
    server_runtime: SimulationServerRuntime
    initial_model_revision: str
    initial_validation: SimulationEvaluation
    active: ActiveSimulationState
    completed_rounds: tuple[SimulationRoundSummary, ...] = ()
    peer_context_state: PeerContextSimulationState = field(
        default_factory=PeerContextSimulationState
    )
    client_partition_sync_state: ClientPartitionSyncSimulationState = field(
        default_factory=ClientPartitionSyncSimulationState
    )
    query_ssl_algorithm_sync_state: QuerySslAlgorithmSyncSimulationState = field(
        default_factory=QuerySslAlgorithmSyncSimulationState
    )
    peer_probe_rows: tuple[LabeledQueryRow, ...] = ()
    peer_probe_manifest: FederatedPeerProbeManifest | None = None
    runtime_resource_cache: RuntimeResourceCache | None = None
    round_base_snapshot_cache: RoundBaseSnapshotCache | None = None


@dataclass(frozen=True, slots=True)
class ClientRoundExecution:
    """client shard 하나의 local training 실행 결과."""

    summary: ClientRoundSummary
    update_submitted: bool
    peer_client_snapshot: FederatedSslPeerClientSnapshot | None = None
    client_partition_snapshot: Mapping[str, Any] = field(default_factory=dict)
    query_ssl_algorithm_state: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RoundExecution:
    """round 실행 후 새 active 상태와 summary."""

    active: ActiveSimulationState
    peer_context_state: PeerContextSimulationState
    client_partition_sync_state: ClientPartitionSyncSimulationState
    query_ssl_algorithm_sync_state: QuerySslAlgorithmSyncSimulationState
    summary: SimulationRoundSummary | None

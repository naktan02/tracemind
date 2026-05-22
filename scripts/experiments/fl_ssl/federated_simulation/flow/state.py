"""FL simulation 내부 실행 context 모델."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from methods.federated_ssl.peer_context import FederatedSslPeerClientSnapshot
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    FederatedClientShard,
    FederatedDatasetSplit,
    SimulationEvaluation,
    SimulationRoundSummary,
)
from scripts.runtime_adapters.federated_server.runtime import SimulationServerRuntime
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
class BootstrappedSimulation:
    """bootstrap 이후 FL simulation loop에 필요한 context."""

    dataset_split: FederatedDatasetSplit
    validation_client_shards: tuple[FederatedClientShard, ...]
    server_runtime: SimulationServerRuntime
    initial_model_revision: str
    initial_validation: SimulationEvaluation
    active: ActiveSimulationState


@dataclass(frozen=True, slots=True)
class ClientRoundExecution:
    """client shard 하나의 local training 실행 결과."""

    summary: ClientRoundSummary
    update_submitted: bool
    peer_client_snapshot: FederatedSslPeerClientSnapshot | None = None


@dataclass(frozen=True, slots=True)
class RoundExecution:
    """round 실행 후 새 active 상태와 summary."""

    active: ActiveSimulationState
    peer_context_state: PeerContextSimulationState
    summary: SimulationRoundSummary | None

"""FL simulation 내부 실행 context 모델."""

from __future__ import annotations

from dataclasses import dataclass

from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    FederatedClientShard,
    FederatedDatasetSplit,
    SimulationEvaluation,
    SimulationRoundSummary,
)
from scripts.runtime_adapters.federated_server.runtime import SimulationServerRuntime
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter


@dataclass(frozen=True, slots=True)
class ActiveSimulationState:
    """현재 round에서 client가 읽는 active global 상태."""

    manifest: ModelManifest
    adapter_state: SharedAdapterState
    prototype_pack: PrototypePackPayload


@dataclass(frozen=True, slots=True)
class BootstrappedSimulation:
    """bootstrap 이후 FL simulation loop에 필요한 context."""

    dataset_split: FederatedDatasetSplit
    validation_client_shards: tuple[FederatedClientShard, ...]
    adapter: EmbeddingAdapter
    server_runtime: SimulationServerRuntime
    initial_model_revision: str
    initial_prototype_version: str
    initial_validation: SimulationEvaluation
    active: ActiveSimulationState


@dataclass(frozen=True, slots=True)
class ClientRoundExecution:
    """client shard 하나의 local training 실행 결과."""

    summary: ClientRoundSummary
    update_submitted: bool


@dataclass(frozen=True, slots=True)
class RoundExecution:
    """round 실행 후 새 active 상태와 summary."""

    active: ActiveSimulationState
    summary: SimulationRoundSummary | None

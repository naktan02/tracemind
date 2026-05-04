"""Federated SSL method runtime seam."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.training.examples.models import EmbeddedTrainingExample
from agent.src.services.training.execution.local_training_service import (
    LocalTrainingService,
)
from main_server.src.services.federation.rounds.boundary.models import (
    RoundOpenRequest,
)
from methods.federated_ssl.base import FederatedSslMethodDescriptor
from scripts.experiments.federated_simulation.models import (
    FederatedTrainingTaskConfig,
)
from scripts.labeled_query_rows import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.training.shared_adapter_state import (
    SharedAdapterState,
)
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter


class FederatedSslMethodRuntime(Protocol):
    """Simulation loop가 호출하는 FL SSL method 실행 조합."""

    descriptor: FederatedSslMethodDescriptor

    def build_round_open_request(
        self,
        *,
        active_manifest: ModelManifest,
        round_id: str,
        training_task_config: FederatedTrainingTaskConfig,
    ) -> RoundOpenRequest:
        """method별 round task를 생성한다."""

    def build_training_examples(
        self,
        *,
        rows: list[LabeledQueryRow],
        adapter: EmbeddingAdapter,
        adapter_state: SharedAdapterState,
        prototype_pack: PrototypePackPayload,
        model_id: str,
        scoring_service: ScoringService,
        objective_config: TrainingObjectiveConfig,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        """client shard row를 method별 local training 입력으로 변환한다."""

    def build_local_training_service(
        self,
        *,
        client_state_root: Path,
    ) -> LocalTrainingService:
        """client local trainer를 생성한다."""

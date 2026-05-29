"""Local update backend protocol for adaptation methods."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingTask,
)
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

if TYPE_CHECKING:
    from shared.src.domain.entities.inference.events import ScoredEvent
    from shared.src.domain.entities.training.pseudo_label_candidate import (
        PseudoLabelCandidate,
    )


class AcceptedTrainingExample(Protocol):
    """Local update backend가 필요로 하는 accepted example 최소 surface."""

    update_embedding: list[float]
    update_scored_event: "ScoredEvent"
    candidate: "PseudoLabelCandidate | None"


class SharedAdapterTrainingBackend(Protocol):
    """Accepted local examples를 shared adapter update로 바꾸는 backend."""

    backend_name: str
    payload_format: str
    adapter_kind: str

    def build_update(
        self,
        *,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        accepted_examples: tuple[AcceptedTrainingExample, ...],
        created_at: datetime,
    ) -> SharedAdapterUpdate:
        """accepted local examples를 기반으로 shared adapter update를 계산한다."""

    def to_payload(self, update: SharedAdapterUpdate) -> SharedAdapterUpdatePayload:
        """저장 가능한 payload 형식으로 변환한다."""

    def build_client_metrics(self, update: SharedAdapterUpdate) -> dict[str, float]:
        """Envelope.client_metrics에 기록할 backend별 요약 metric을 만든다."""

    def matches_objective_config(
        self,
        objective_config: TrainingObjectiveConfig | None,
    ) -> bool:
        """현재 인스턴스를 objective config에 재사용할 수 있는지 판단한다."""


TrainingBackend = SharedAdapterTrainingBackend
TrainingBackendFactory = Callable[
    [TrainingObjectiveConfig | None],
    SharedAdapterTrainingBackend,
]

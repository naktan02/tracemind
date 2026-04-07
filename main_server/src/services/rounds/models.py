"""FL round lifecycle domain 모델."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    SecureAggregationConfig,
    TrainingConfigScalar,
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
    TrainingUpdateEnvelope,
)


class RoundStatus(StrEnum):
    """FL round 상태."""

    OPEN = "open"
    FINALIZED = "finalized"


@dataclass(slots=True)
class RoundPublicationSummary:
    """라운드 finalize 이후 남기는 publication 요약."""

    next_manifest: ModelManifest
    aggregated_metrics: dict[str, float]
    update_count: int
    finalized_at: datetime
    prototype_pack_ref: str | None = None
    prototype_build_state_ref: str | None = None
    prototype_rebuild_input_id: str | None = None


@dataclass(slots=True)
class RoundRecord:
    """라운드 하나의 canonical runtime 상태."""

    round_id: str
    status: RoundStatus
    active_manifest: ModelManifest
    training_task: TrainingTask
    created_at: datetime
    updated_at: datetime
    updates: tuple[TrainingUpdateEnvelope, ...] = field(default_factory=tuple)
    finalized_at: datetime | None = None
    publication: RoundPublicationSummary | None = None


@dataclass(slots=True)
class RoundOpenRequest:
    """새 round open 요청."""

    active_manifest: ModelManifest
    round_id: str | None = None
    task_id: str | None = None
    task_type: TrainingTaskType = TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING
    local_epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-4
    max_steps: int = 50
    objective_config: (
        TrainingObjectiveConfig | Mapping[str, TrainingConfigScalar] | None
    ) = None
    selection_policy: (
        TrainingSelectionPolicy | Mapping[str, TrainingConfigScalar] | None
    ) = None
    secure_aggregation: (
        SecureAggregationConfig | Mapping[str, TrainingConfigScalar] | bool | None
    ) = None
    min_required_examples: int | None = None
    gradient_clip_norm: float | None = None
    deadline_at: datetime | None = None
    notes: str | None = None


@dataclass(slots=True)
class RoundFinalizeRequest:
    """round finalize 요청."""

    next_prototype_version: str
    next_model_revision: str | None = None
    published_at: datetime | None = None


@dataclass(slots=True)
class RoundUpdateAcceptance:
    """update 등록 성공 응답."""

    round_id: str
    update_id: str
    update_count: int
    accepted_at: datetime
    idempotent: bool = False

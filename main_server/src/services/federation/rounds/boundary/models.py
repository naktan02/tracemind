"""FL round lifecycle domain 모델."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from methods.federated_ssl.runtime_fallbacks import RUNTIME_FALLBACK_TRAINING_PROFILE
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
    round_state_summary_metrics: dict[str, float] = field(default_factory=dict)
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


@dataclass(slots=True, kw_only=True)
class RoundTaskConfig:
    """active manifest를 제외한 reusable round task 설정."""

    task_type: TrainingTaskType = TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING
    local_epochs: int = RUNTIME_FALLBACK_TRAINING_PROFILE.local_epochs
    batch_size: int = RUNTIME_FALLBACK_TRAINING_PROFILE.batch_size
    learning_rate: float = RUNTIME_FALLBACK_TRAINING_PROFILE.learning_rate
    max_steps: int = RUNTIME_FALLBACK_TRAINING_PROFILE.max_steps
    objective_config: (
        TrainingObjectiveConfig | Mapping[str, TrainingConfigScalar] | None
    ) = None
    selection_policy: (
        TrainingSelectionPolicy | Mapping[str, TrainingConfigScalar] | None
    ) = None
    secure_aggregation: (
        SecureAggregationConfig | Mapping[str, TrainingConfigScalar] | bool | None
    ) = None
    min_required_examples: int | None = (
        RUNTIME_FALLBACK_TRAINING_PROFILE.min_required_examples
    )
    gradient_clip_norm: float | None = (
        RUNTIME_FALLBACK_TRAINING_PROFILE.gradient_clip_norm
    )
    deadline_at: datetime | None = None
    notes: str | None = None

    def to_round_open_request(
        self,
        *,
        active_manifest: ModelManifest,
        round_id: str | None = None,
        task_id: str | None = None,
    ) -> "RoundOpenRequest":
        """reusable task 설정을 round open request로 변환한다."""

        return RoundOpenRequest(
            active_manifest=active_manifest,
            round_id=round_id,
            task_id=task_id,
            task_type=self.task_type,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            objective_config=self.objective_config,
            selection_policy=self.selection_policy,
            secure_aggregation=self.secure_aggregation,
            min_required_examples=self.min_required_examples,
            gradient_clip_norm=self.gradient_clip_norm,
            deadline_at=self.deadline_at,
            notes=self.notes,
        )


@dataclass(slots=True, kw_only=True)
class RoundOpenDraftRequest(RoundTaskConfig):
    """active manifest를 아직 붙이지 않은 새 round open 요청."""

    round_id: str | None = None
    task_id: str | None = None


@dataclass(slots=True, kw_only=True)
class RoundOpenRequest(RoundTaskConfig):
    """서버 active manifest로 resolve된 새 round open 요청."""

    active_manifest: ModelManifest
    round_id: str | None = None
    task_id: str | None = None


@dataclass(slots=True)
class RoundFinalizeRequest:
    """round finalize 요청."""

    next_model_revision: str | None = None
    next_auxiliary_artifact_versions: Mapping[str, str] = field(default_factory=dict)
    published_at: datetime | None = None


@dataclass(slots=True)
class RoundUpdateAcceptance:
    """update 등록 성공 응답."""

    round_id: str
    update_id: str
    update_count: int
    accepted_at: datetime
    idempotent: bool = False

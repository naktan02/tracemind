"""FL round의 task 발행과 pair publication을 조정한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from shared.src.contracts.adapter_contracts import load_shared_adapter_update_payload
from shared.src.domain.entities.artifacts.model_manifest import ModelManifest
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.training_task import TrainingTask
from shared.src.domain.entities.training.training_task_config import (
    TrainingConfigScalar,
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)
from shared.src.domain.entities.training.training_update import TrainingUpdateEnvelope
from src.infrastructure.repositories.vector_adapter_state_repository import (
    SharedAdapterStateRepository,
)
from src.services.rounds.adapter_family_service import (
    DiagonalScaleRoundFamily,
    SharedAdapterRoundFamily,
)


@dataclass(slots=True)
class RoundPublication:
    """한 라운드 집계 후 발행되는 새 active pair 메타데이터."""

    next_manifest: ModelManifest
    next_state: SharedAdapterState
    aggregated_metrics: dict[str, float]
    update_count: int


@dataclass(slots=True)
class TrainingTaskRequest:
    """라운드 학습 task 생성 요청."""

    active_manifest: ModelManifest
    round_id: str
    task_id: str | None = None
    task_type: str = "pseudo_label_self_training"
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
    min_required_examples: int | None = None
    gradient_clip_norm: float | None = None
    deadline_at: datetime | None = None
    notes: str | None = None


@dataclass(slots=True)
class RoundPublicationRequest:
    """집계 후 active pair 발행 요청."""

    base_manifest: ModelManifest
    updates: tuple[TrainingUpdateEnvelope, ...] | list[TrainingUpdateEnvelope]
    next_prototype_version: str
    next_model_revision: str | None = None
    published_at: datetime | None = None


@dataclass(slots=True)
class RoundManagerService:
    """라운드용 task를 만들고 새 model/prototype pair를 발행한다."""

    adapter_family: SharedAdapterRoundFamily = field(
        default_factory=DiagonalScaleRoundFamily
    )
    artifact_repository: SharedAdapterStateRepository = field(
        default_factory=SharedAdapterStateRepository
    )

    def create_training_task(self, request: TrainingTaskRequest) -> TrainingTask:
        return TrainingTask(
            schema_version="training_task.v1",
            task_id=request.task_id or f"task_{request.round_id}_{uuid4().hex[:8]}",
            round_id=request.round_id,
            model_id=request.active_manifest.model_id,
            model_revision=request.active_manifest.model_revision,
            task_type=request.task_type,
            training_scope=request.active_manifest.training_scope,
            local_epochs=request.local_epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            max_steps=request.max_steps,
            objective_config=self._resolve_objective_config(request.objective_config),
            selection_policy=self._resolve_selection_policy(request.selection_policy),
            deadline_at=request.deadline_at,
            gradient_clip_norm=request.gradient_clip_norm,
            min_required_examples=request.min_required_examples,
            secure_aggregation_required=False,
            notes=request.notes,
        )

    def publish_next_pair(self, request: RoundPublicationRequest) -> RoundPublication:
        if not request.updates:
            raise ValueError(
                "At least one update is required to publish the next pair."
            )

        effective_published_at = request.published_at or datetime.now(timezone.utc)
        base_state = self._load_base_state(request.base_manifest)
        next_revision = (
            request.next_model_revision
            or f"{request.base_manifest.model_revision}_next"
        )
        update_payloads = [
            self._load_update_payload(update) for update in request.updates
        ]
        aggregation = self.adapter_family.aggregation_backend.aggregate(
            base_state=base_state,
            update_payloads=update_payloads,
            next_model_revision=next_revision,
            aggregated_at=effective_published_at,
        )
        artifact_path = self.artifact_repository.save_shared_adapter_state(
            self.adapter_family.state_to_payload(aggregation.next_state)
        )
        next_manifest = ModelManifest(
            schema_version=request.base_manifest.schema_version,
            model_id=request.base_manifest.model_id,
            model_revision=aggregation.next_state.model_revision,
            published_at=effective_published_at,
            artifact_kind="shared_adapter_state",
            artifact_ref=str(artifact_path),
            prototype_version=request.next_prototype_version,
            training_scope=request.base_manifest.training_scope,
            training_enabled=request.base_manifest.training_enabled,
            compatible_task_types=request.base_manifest.compatible_task_types,
            base_model_id=request.base_manifest.base_model_id,
            base_model_revision=request.base_manifest.base_model_revision,
            translation_model_id=request.base_manifest.translation_model_id,
            translation_model_revision=request.base_manifest.translation_model_revision,
            notes=(
                "round_active_pair_only published from "
                f"{request.updates[0].round_id}"
            ),
        )
        return RoundPublication(
            next_manifest=next_manifest,
            next_state=aggregation.next_state,
            aggregated_metrics=aggregation.aggregated_metrics,
            update_count=aggregation.update_count,
        )

    def _load_base_state(self, base_manifest: ModelManifest) -> SharedAdapterState:
        payload = self.artifact_repository.load_shared_adapter_state_from_ref(
            base_manifest.artifact_ref
        )
        state = self.adapter_family.state_from_payload(payload)
        if state.adapter_kind != self.adapter_family.aggregation_backend.adapter_kind:
            raise ValueError(
                "Base state adapter_kind does not match the configured "
                f"aggregation backend: {state.adapter_kind}"
            )
        return state

    def _load_update_payload(self, update: TrainingUpdateEnvelope):
        payload = load_shared_adapter_update_payload(Path(update.payload_ref))
        if update.payload_format not in self.adapter_family.accepted_update_formats:
            raise ValueError(
                "Unsupported payload_format for adapter_kind "
                f"{payload.adapter_kind}: {update.payload_format}"
            )
        return self.adapter_family.update_from_payload(payload)

    @staticmethod
    def _resolve_objective_config(
        source: TrainingObjectiveConfig | Mapping[str, TrainingConfigScalar] | None,
    ) -> TrainingObjectiveConfig:
        if isinstance(source, TrainingObjectiveConfig):
            return source
        return TrainingObjectiveConfig.from_mapping(
            source
            or {
                "loss": "diagonal_scale_heuristic",
                "confidence_threshold": 0.6,
                "margin_threshold": 0.02,
                "score_policy_name": "max_cosine",
                "acceptance_policy_name": "top1_margin_threshold",
                "privacy_guard_name": "diagonal_scale_clip_only",
            }
        )

    @staticmethod
    def _resolve_selection_policy(
        source: TrainingSelectionPolicy | Mapping[str, TrainingConfigScalar] | None,
    ) -> TrainingSelectionPolicy:
        if isinstance(source, TrainingSelectionPolicy):
            return source
        return TrainingSelectionPolicy.from_mapping(
            source or {"max_examples": 128}
        )

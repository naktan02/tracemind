"""FL round의 task 발행과 pair publication을 조정한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from shared.src.contracts.adapter_contracts import (
    VectorAdapterStatePayload,
    load_vector_adapter_delta_payload,
)
from shared.src.domain.entities.artifacts.model_manifest import ModelManifest
from shared.src.domain.entities.training.training_task import TrainingTask
from shared.src.domain.entities.training.training_update import TrainingUpdateEnvelope
from shared.src.domain.entities.training.vector_adapter_delta import (
    VectorAdapterDelta,
)
from shared.src.domain.entities.training.vector_adapter_state import (
    VectorAdapterState,
)
from src.infrastructure.repositories.vector_adapter_state_repository import (
    VectorAdapterStateRepository,
)
from src.services.rounds.aggregation_service import AggregationService


@dataclass(slots=True)
class RoundPublication:
    """한 라운드 집계 후 발행되는 새 active pair 메타데이터."""

    next_manifest: ModelManifest
    next_state: VectorAdapterState
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
    objective_config: dict[str, str | int | float | bool] | None = None
    selection_policy: dict[str, str | int | float | bool] | None = None
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

    aggregation_service: AggregationService = field(default_factory=AggregationService)
    artifact_repository: VectorAdapterStateRepository = field(
        default_factory=VectorAdapterStateRepository
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
            objective_config=request.objective_config
            or {
                "loss": "synthetic_vector_adapter",
                "confidence_threshold": 0.6,
                "margin_threshold": 0.02,
            },
            selection_policy=request.selection_policy or {"max_examples": 128},
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
        aggregation = self.aggregation_service.aggregate(
            base_state=base_state,
            update_payloads=update_payloads,
            next_model_revision=next_revision,
            aggregated_at=effective_published_at,
        )
        artifact_path = self.artifact_repository.save_state(
            self._to_state_payload(aggregation.next_state)
        )
        next_manifest = ModelManifest(
            schema_version=request.base_manifest.schema_version,
            model_id=request.base_manifest.model_id,
            model_revision=aggregation.next_state.model_revision,
            published_at=effective_published_at,
            artifact_kind=request.base_manifest.artifact_kind,
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

    def _load_base_state(self, base_manifest: ModelManifest) -> VectorAdapterState:
        payload = self.artifact_repository.load_state_from_ref(
            base_manifest.artifact_ref
        )
        return VectorAdapterState(
            schema_version=payload.schema_version,
            model_id=payload.model_id,
            model_revision=payload.model_revision,
            training_scope=payload.training_scope,
            dimension_scales=list(payload.dimension_scales),
            updated_at=payload.updated_at,
        )

    @staticmethod
    def _load_update_payload(update: TrainingUpdateEnvelope) -> VectorAdapterDelta:
        payload = load_vector_adapter_delta_payload(Path(update.payload_ref))
        return VectorAdapterDelta(
            schema_version=payload.schema_version,
            model_id=payload.model_id,
            base_model_revision=payload.base_model_revision,
            training_scope=payload.training_scope,
            dimension_deltas=list(payload.dimension_deltas),
            example_count=payload.example_count,
            mean_confidence=payload.mean_confidence,
            created_at=payload.created_at,
            mean_margin=payload.mean_margin,
            label_counts=dict(payload.label_counts),
        )

    @staticmethod
    def _to_state_payload(state: VectorAdapterState) -> VectorAdapterStatePayload:
        return VectorAdapterStatePayload(
            schema_version=state.schema_version,
            model_id=state.model_id,
            model_revision=state.model_revision,
            training_scope=state.training_scope,
            dimension_scales=state.dimension_scales,
            updated_at=state.updated_at,
        )

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from shared.src.contracts.adapter_contracts import (
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
)
from shared.src.contracts.common_types import (
    TrainingScope,
    TrainingTaskType,
)
from shared.src.contracts.model_contracts import (
    ArtifactKind,
    ModelManifest,
    ModelManifestPayload,
)
from shared.src.contracts.personalization_contracts import (
    PersonalizationStatePayload,
    PersonalizationWarmupStatus,
)
from shared.src.contracts.training_contracts import (
    DecisionFeedbackSignalPayload,
    FeedbackSignalType,
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
    TrainingTaskPayload,
    TrainingUpdateEnvelopePayload,
    UpdatePayloadFormat,
)

FIXED_UTC_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def fixed_utc_time() -> datetime:
    """shared 테스트 전반에서 재사용하는 고정 UTC 시각."""
    return FIXED_UTC_TIME


@pytest.fixture
def make_model_manifest_payload(
    fixed_utc_time: datetime,
) -> Callable[..., ModelManifestPayload]:
    def _make(**overrides: object) -> ModelManifestPayload:
        defaults = {
            "model_id": "tracemind-embed",
            "model_revision": "rev_001",
            "published_at": fixed_utc_time,
            "artifact_kind": ArtifactKind.SHARED_ADAPTER_STATE,
            "artifact_ref": "/tmp/rev_001.json",
            "prototype_version": "proto_001",
            "training_scope": TrainingScope.ADAPTER_ONLY,
            "training_enabled": True,
            "compatible_task_types": (
                TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,
            ),
        }
        return ModelManifestPayload(**(defaults | overrides))

    return _make


@pytest.fixture
def make_model_manifest(
    make_model_manifest_payload,
) -> Callable[..., ModelManifest]:
    def _make(**overrides: object) -> ModelManifest:
        return make_model_manifest_payload(**overrides)

    return _make


@pytest.fixture
def make_adapter_state_payload(
    fixed_utc_time: datetime,
) -> Callable[..., DiagonalScaleAdapterStatePayload]:
    def _make(**overrides: object) -> DiagonalScaleAdapterStatePayload:
        defaults = {
            "model_id": "tracemind-embed",
            "model_revision": "rev_001",
            "training_scope": TrainingScope.ADAPTER_ONLY,
            "dimension_scales": [1.0, 0.95],
            "updated_at": fixed_utc_time,
        }
        return DiagonalScaleAdapterStatePayload(**(defaults | overrides))

    return _make


@pytest.fixture
def make_adapter_update_payload() -> Callable[..., DiagonalScaleAdapterUpdatePayload]:
    def _make(**overrides: object) -> DiagonalScaleAdapterUpdatePayload:
        defaults = {
            "model_id": "tracemind-embed",
            "base_model_revision": "rev_001",
            "training_scope": TrainingScope.ADAPTER_ONLY,
            "dimension_deltas": [0.01, -0.02],
            "example_count": 8,
            "mean_confidence": 0.84,
            "mean_margin": 0.12,
            "label_counts": {"anxiety": 5, "depression": 3},
        }
        return DiagonalScaleAdapterUpdatePayload(**(defaults | overrides))

    return _make


@pytest.fixture
def make_training_task_payload() -> Callable[..., TrainingTaskPayload]:
    def _make(**overrides: object) -> TrainingTaskPayload:
        defaults = {
            "task_id": "task_001",
            "round_id": "round_001",
            "model_id": "tracemind-embed",
            "model_revision": "rev_001",
            "task_type": TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,
            "training_scope": TrainingScope.ADAPTER_ONLY,
            "local_epochs": 1,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "max_steps": 10,
            "objective_config": TrainingObjectiveConfigPayload(
                training_backend_name="contrastive"
            ),
            "selection_policy": TrainingSelectionPolicyPayload(max_examples=32),
        }
        return TrainingTaskPayload(**(defaults | overrides))

    return _make


@pytest.fixture
def make_training_update_envelope_payload(
    fixed_utc_time: datetime,
) -> Callable[..., TrainingUpdateEnvelopePayload]:
    def _make(**overrides: object) -> TrainingUpdateEnvelopePayload:
        defaults = {
            "update_id": "update_001",
            "round_id": "round_001",
            "task_id": "task_001",
            "model_id": "tracemind-embed",
            "base_model_revision": "rev_001",
            "training_scope": TrainingScope.ADAPTER_ONLY,
            "payload_ref": "updates/update_001",
            "payload_format": UpdatePayloadFormat.DIAGONAL_SCALE_UPDATE,
            "example_count": 12,
            "client_metrics": {"mean_loss": 0.5},
            "created_at": fixed_utc_time,
        }
        return TrainingUpdateEnvelopePayload(**(defaults | overrides))

    return _make


@pytest.fixture
def make_feedback_signal_payload(
    fixed_utc_time: datetime,
) -> Callable[..., DecisionFeedbackSignalPayload]:
    def _make(**overrides: object) -> DecisionFeedbackSignalPayload:
        defaults = {
            "signal_id": "signal_001",
            "signal_type": FeedbackSignalType.PSEUDO_LABEL,
            "label": "depression_rising",
            "confidence": 0.9,
            "occurred_at": fixed_utc_time,
        }
        return DecisionFeedbackSignalPayload(**(defaults | overrides))

    return _make


@pytest.fixture
def make_personalization_state_payload(
    fixed_utc_time: datetime,
) -> Callable[..., PersonalizationStatePayload]:
    def _make(**overrides: object) -> PersonalizationStatePayload:
        defaults = {
            "state_version": "ps_001",
            "baseline_by_category": {"depression": 0.2},
            "threshold_by_category": {"depression": 0.6},
            "warmup_status": PersonalizationWarmupStatus.READY,
            "updated_at": fixed_utc_time,
        }
        return PersonalizationStatePayload(**(defaults | overrides))

    return _make

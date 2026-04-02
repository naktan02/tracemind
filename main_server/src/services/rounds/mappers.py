"""FL round domain <-> payload 변환 유틸리티."""

from __future__ import annotations

from shared.src.contracts.model_contracts import ModelManifestPayload
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
    TrainingTaskPayload,
    TrainingUpdateEnvelopePayload,
)
from shared.src.domain.entities.artifacts.model_manifest import ModelManifest
from shared.src.domain.entities.training.training_task import TrainingTask
from shared.src.domain.entities.training.training_task_config import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)
from shared.src.domain.entities.training.training_update import TrainingUpdateEnvelope

from .models import (
    RoundFinalizeRequest,
    RoundOpenRequest,
    RoundPublicationSummary,
    RoundRecord,
    RoundUpdateAcceptance,
)
from .payloads import (
    RoundFinalizeRequestPayload,
    RoundOpenRequestPayload,
    RoundPublicationPayload,
    RoundRecordPayload,
    RoundUpdateAcceptancePayload,
)


def model_manifest_to_payload(manifest: ModelManifest) -> ModelManifestPayload:
    """domain manifest를 payload로 변환한다."""
    return ModelManifestPayload(
        schema_version=manifest.schema_version,
        model_id=manifest.model_id,
        model_revision=manifest.model_revision,
        published_at=manifest.published_at,
        artifact_kind=manifest.artifact_kind,
        artifact_ref=manifest.artifact_ref,
        prototype_version=manifest.prototype_version,
        training_scope=manifest.training_scope,
        training_enabled=manifest.training_enabled,
        compatible_task_types=list(manifest.compatible_task_types),
        base_model_id=manifest.base_model_id,
        base_model_revision=manifest.base_model_revision,
        translation_model_id=manifest.translation_model_id,
        translation_model_revision=manifest.translation_model_revision,
        notes=manifest.notes,
    )


def model_manifest_from_payload(payload: ModelManifestPayload) -> ModelManifest:
    """payload manifest를 domain으로 변환한다."""
    return ModelManifest(
        schema_version=payload.schema_version,
        model_id=payload.model_id,
        model_revision=payload.model_revision,
        published_at=payload.published_at,
        artifact_kind=payload.artifact_kind,
        artifact_ref=payload.artifact_ref,
        prototype_version=payload.prototype_version,
        training_scope=payload.training_scope,
        training_enabled=payload.training_enabled,
        compatible_task_types=tuple(payload.compatible_task_types),
        base_model_id=payload.base_model_id,
        base_model_revision=payload.base_model_revision,
        translation_model_id=payload.translation_model_id,
        translation_model_revision=payload.translation_model_revision,
        notes=payload.notes,
    )


def training_task_to_payload(task: TrainingTask) -> TrainingTaskPayload:
    """domain training task를 payload로 변환한다."""
    return TrainingTaskPayload(
        schema_version=task.schema_version,
        task_id=task.task_id,
        round_id=task.round_id,
        model_id=task.model_id,
        model_revision=task.model_revision,
        task_type=task.task_type,
        training_scope=task.training_scope,
        local_epochs=task.local_epochs,
        batch_size=task.batch_size,
        learning_rate=task.learning_rate,
        max_steps=task.max_steps,
        objective_config=TrainingObjectiveConfigPayload(
            loss=task.objective_config.loss,
            confidence_threshold=task.objective_config.confidence_threshold,
            margin_threshold=task.objective_config.margin_threshold,
            score_policy_name=task.objective_config.score_policy_name,
            score_top_k=task.objective_config.score_top_k,
            acceptance_policy_name=task.objective_config.acceptance_policy_name,
            privacy_guard_name=task.objective_config.privacy_guard_name,
            extras=dict(task.objective_config.extras),
        ),
        selection_policy=TrainingSelectionPolicyPayload(
            max_examples=task.selection_policy.max_examples,
            require_feedback=task.selection_policy.require_feedback,
            extras=dict(task.selection_policy.extras),
        ),
        deadline_at=task.deadline_at,
        gradient_clip_norm=task.gradient_clip_norm,
        min_required_examples=task.min_required_examples,
        secure_aggregation_required=task.secure_aggregation_required,
        notes=task.notes,
    )


def training_task_from_payload(payload: TrainingTaskPayload) -> TrainingTask:
    """payload training task를 domain으로 변환한다."""
    return TrainingTask(
        schema_version=payload.schema_version,
        task_id=payload.task_id,
        round_id=payload.round_id,
        model_id=payload.model_id,
        model_revision=payload.model_revision,
        task_type=payload.task_type,
        training_scope=payload.training_scope,
        local_epochs=payload.local_epochs,
        batch_size=payload.batch_size,
        learning_rate=payload.learning_rate,
        max_steps=payload.max_steps,
        objective_config=TrainingObjectiveConfig(
            loss=payload.objective_config.loss,
            confidence_threshold=payload.objective_config.confidence_threshold,
            margin_threshold=payload.objective_config.margin_threshold,
            score_policy_name=payload.objective_config.score_policy_name,
            score_top_k=payload.objective_config.score_top_k,
            acceptance_policy_name=payload.objective_config.acceptance_policy_name,
            privacy_guard_name=payload.objective_config.privacy_guard_name,
            extras=dict(payload.objective_config.extras),
        ),
        selection_policy=TrainingSelectionPolicy(
            max_examples=payload.selection_policy.max_examples,
            require_feedback=payload.selection_policy.require_feedback,
            extras=dict(payload.selection_policy.extras),
        ),
        deadline_at=payload.deadline_at,
        gradient_clip_norm=payload.gradient_clip_norm,
        min_required_examples=payload.min_required_examples,
        secure_aggregation_required=payload.secure_aggregation_required,
        notes=payload.notes,
    )


def training_update_to_payload(
    envelope: TrainingUpdateEnvelope,
) -> TrainingUpdateEnvelopePayload:
    """domain training update를 payload로 변환한다."""
    return TrainingUpdateEnvelopePayload(
        schema_version=envelope.schema_version,
        update_id=envelope.update_id,
        round_id=envelope.round_id,
        task_id=envelope.task_id,
        model_id=envelope.model_id,
        base_model_revision=envelope.base_model_revision,
        training_scope=envelope.training_scope,
        payload_ref=envelope.payload_ref,
        payload_format=envelope.payload_format,
        example_count=envelope.example_count,
        client_metrics=dict(envelope.client_metrics),
        created_at=envelope.created_at,
        clipped=envelope.clipped,
        dp_applied=envelope.dp_applied,
        checksum=envelope.checksum,
        notes=envelope.notes,
    )


def training_update_from_payload(
    payload: TrainingUpdateEnvelopePayload,
) -> TrainingUpdateEnvelope:
    """payload training update를 domain으로 변환한다."""
    return TrainingUpdateEnvelope(
        schema_version=payload.schema_version,
        update_id=payload.update_id,
        round_id=payload.round_id,
        task_id=payload.task_id,
        model_id=payload.model_id,
        base_model_revision=payload.base_model_revision,
        training_scope=payload.training_scope,
        payload_ref=payload.payload_ref,
        payload_format=payload.payload_format,
        example_count=payload.example_count,
        client_metrics=dict(payload.client_metrics),
        created_at=payload.created_at,
        clipped=payload.clipped,
        dp_applied=payload.dp_applied,
        checksum=payload.checksum,
        notes=payload.notes,
    )


def round_publication_to_payload(
    publication: RoundPublicationSummary,
) -> RoundPublicationPayload:
    """domain publication summary를 payload로 변환한다."""
    return RoundPublicationPayload(
        next_manifest=model_manifest_to_payload(publication.next_manifest),
        aggregated_metrics=dict(publication.aggregated_metrics),
        update_count=publication.update_count,
        finalized_at=publication.finalized_at,
        prototype_pack_ref=publication.prototype_pack_ref,
        prototype_build_state_ref=publication.prototype_build_state_ref,
        prototype_rebuild_input_id=publication.prototype_rebuild_input_id,
    )


def round_publication_from_payload(
    payload: RoundPublicationPayload,
) -> RoundPublicationSummary:
    """payload publication summary를 domain으로 변환한다."""
    return RoundPublicationSummary(
        next_manifest=model_manifest_from_payload(payload.next_manifest),
        aggregated_metrics=dict(payload.aggregated_metrics),
        update_count=payload.update_count,
        finalized_at=payload.finalized_at,
        prototype_pack_ref=payload.prototype_pack_ref,
        prototype_build_state_ref=payload.prototype_build_state_ref,
        prototype_rebuild_input_id=payload.prototype_rebuild_input_id,
    )


def round_record_to_payload(record: RoundRecord) -> RoundRecordPayload:
    """domain round record를 payload로 변환한다."""
    return RoundRecordPayload(
        round_id=record.round_id,
        status=record.status,
        active_manifest=model_manifest_to_payload(record.active_manifest),
        training_task=training_task_to_payload(record.training_task),
        updates=[training_update_to_payload(update) for update in record.updates],
        created_at=record.created_at,
        updated_at=record.updated_at,
        finalized_at=record.finalized_at,
        publication=(
            round_publication_to_payload(record.publication)
            if record.publication is not None
            else None
        ),
    )


def round_record_from_payload(payload: RoundRecordPayload) -> RoundRecord:
    """payload round record를 domain으로 변환한다."""
    return RoundRecord(
        round_id=payload.round_id,
        status=payload.status,
        active_manifest=model_manifest_from_payload(payload.active_manifest),
        training_task=training_task_from_payload(payload.training_task),
        updates=tuple(
            training_update_from_payload(update_payload)
            for update_payload in payload.updates
        ),
        created_at=payload.created_at,
        updated_at=payload.updated_at,
        finalized_at=payload.finalized_at,
        publication=(
            round_publication_from_payload(payload.publication)
            if payload.publication is not None
            else None
        ),
    )


def round_open_request_from_payload(
    payload: RoundOpenRequestPayload,
) -> RoundOpenRequest:
    """API payload를 domain open request로 변환한다."""
    return RoundOpenRequest(
        active_manifest=model_manifest_from_payload(payload.active_manifest),
        round_id=payload.round_id,
        task_id=payload.task_id,
        task_type=payload.task_type,
        local_epochs=payload.local_epochs,
        batch_size=payload.batch_size,
        learning_rate=payload.learning_rate,
        max_steps=payload.max_steps,
        objective_config=(
            TrainingObjectiveConfig(
                loss=payload.objective_config.loss,
                confidence_threshold=payload.objective_config.confidence_threshold,
                margin_threshold=payload.objective_config.margin_threshold,
                score_policy_name=payload.objective_config.score_policy_name,
                score_top_k=payload.objective_config.score_top_k,
                acceptance_policy_name=payload.objective_config.acceptance_policy_name,
                privacy_guard_name=payload.objective_config.privacy_guard_name,
                extras=dict(payload.objective_config.extras),
            )
            if payload.objective_config is not None
            else None
        ),
        selection_policy=(
            TrainingSelectionPolicy(
                max_examples=payload.selection_policy.max_examples,
                require_feedback=payload.selection_policy.require_feedback,
                extras=dict(payload.selection_policy.extras),
            )
            if payload.selection_policy is not None
            else None
        ),
        min_required_examples=payload.min_required_examples,
        gradient_clip_norm=payload.gradient_clip_norm,
        deadline_at=payload.deadline_at,
        notes=payload.notes,
    )


def round_finalize_request_from_payload(
    payload: RoundFinalizeRequestPayload,
) -> RoundFinalizeRequest:
    """API payload를 domain finalize request로 변환한다."""
    return RoundFinalizeRequest(
        next_prototype_version=payload.next_prototype_version,
        next_model_revision=payload.next_model_revision,
        published_at=payload.published_at,
    )


def round_update_acceptance_to_payload(
    acceptance: RoundUpdateAcceptance,
) -> RoundUpdateAcceptancePayload:
    """domain update acceptance를 API payload로 변환한다."""
    return RoundUpdateAcceptancePayload(
        round_id=acceptance.round_id,
        update_id=acceptance.update_id,
        update_count=acceptance.update_count,
        accepted_at=acceptance.accepted_at,
        idempotent=acceptance.idempotent,
    )

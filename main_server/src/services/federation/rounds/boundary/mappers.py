"""FL round domain <-> payload 변환 유틸리티."""

from __future__ import annotations

from shared.src.contracts.model_contracts import ModelManifest, ModelManifestPayload
from shared.src.contracts.training_contracts import (
    TrainingTask,
    TrainingTaskPayload,
    TrainingUpdateEnvelope,
    TrainingUpdateEnvelopePayload,
)

from .models import (
    RoundFinalizeRequest,
    RoundOpenRequest,
    RoundPublicationSummary,
    RoundRecord,
    RoundTaskConfig,
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
    return manifest


def model_manifest_from_payload(payload: ModelManifestPayload) -> ModelManifest:
    """payload manifest를 domain으로 변환한다."""
    return payload


def training_task_to_payload(task: TrainingTask) -> TrainingTaskPayload:
    """domain training task를 payload로 변환한다."""
    return task


def training_task_from_payload(payload: TrainingTaskPayload) -> TrainingTask:
    """payload training task를 domain으로 변환한다."""
    return payload


def training_update_to_payload(
    envelope: TrainingUpdateEnvelope,
) -> TrainingUpdateEnvelopePayload:
    """domain training update를 payload로 변환한다."""
    return envelope


def training_update_from_payload(
    payload: TrainingUpdateEnvelopePayload,
) -> TrainingUpdateEnvelope:
    """payload training update를 domain으로 변환한다."""
    return payload


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
    return RoundTaskConfig(
        task_type=payload.task_type,
        local_epochs=payload.local_epochs,
        batch_size=payload.batch_size,
        learning_rate=payload.learning_rate,
        max_steps=payload.max_steps,
        objective_config=(
            payload.objective_config if payload.objective_config is not None else None
        ),
        selection_policy=(
            payload.selection_policy if payload.selection_policy is not None else None
        ),
        secure_aggregation=(
            payload.secure_aggregation
            if payload.secure_aggregation is not None
            else None
        ),
        min_required_examples=payload.min_required_examples,
        gradient_clip_norm=payload.gradient_clip_norm,
        deadline_at=payload.deadline_at,
        notes=payload.notes,
    ).to_round_open_request(
        active_manifest=model_manifest_from_payload(payload.active_manifest),
        round_id=payload.round_id,
        task_id=payload.task_id,
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

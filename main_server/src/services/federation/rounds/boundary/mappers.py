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
    InitialSharedArtifactPublicationRequest,
    RoundFinalizeRequest,
    RoundOpenDraftRequest,
    RoundPublicationSummary,
    RoundRecord,
    RoundStrategyConfig,
    RoundUpdateAcceptance,
)
from .payloads import (
    InitialSharedArtifactPublicationRequestPayload,
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
        round_state_summary_metrics=dict(publication.round_state_summary_metrics),
        auxiliary_artifact_refs=dict(publication.auxiliary_artifact_refs),
        auxiliary_artifact_metadata=dict(publication.auxiliary_artifact_metadata),
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
        round_state_summary_metrics=dict(payload.round_state_summary_metrics),
        auxiliary_artifact_refs=dict(payload.auxiliary_artifact_refs),
        auxiliary_artifact_metadata=dict(payload.auxiliary_artifact_metadata),
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


def initial_shared_artifact_publication_request_from_payload(
    payload: InitialSharedArtifactPublicationRequestPayload,
) -> InitialSharedArtifactPublicationRequest:
    """API payload를 initial shared artifact publication request로 변환한다."""

    labels = tuple(str(label).strip() for label in payload.label_schema)
    labels = tuple(label for label in labels if label)
    if not labels:
        raise ValueError("label_schema must contain at least one non-empty label.")
    return InitialSharedArtifactPublicationRequest(
        model_id=payload.model_id,
        model_revision=payload.model_revision,
        training_scope=payload.training_scope,
        label_schema=labels,
        embedding_dim=payload.embedding_dim,
        compatible_task_types=tuple(payload.compatible_task_types),
        notes=payload.notes,
    )


def round_open_draft_request_from_payload(
    payload: RoundOpenRequestPayload,
) -> RoundOpenDraftRequest:
    """API payload를 active manifest 없는 open draft로 변환한다."""
    return RoundOpenDraftRequest(
        task_type=payload.task_type,
        local_epochs=payload.local_epochs,
        batch_size=payload.batch_size,
        learning_rate=payload.learning_rate,
        max_steps=payload.max_steps,
        strategy=(
            RoundStrategyConfig(
                mode=payload.strategy.mode,
                local_update_profile=payload.strategy.local_update_profile,
                ssl_method=payload.strategy.ssl_method,
                fssl_method=payload.strategy.fssl_method,
                scenario=payload.strategy.scenario,
                server_update_policy=payload.strategy.server_update_policy,
                aggregation_backend=payload.strategy.aggregation_backend,
                parameter_overrides=dict(payload.strategy.parameter_overrides),
            )
            if payload.strategy is not None
            else None
        ),
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
        round_id=payload.round_id,
        task_id=payload.task_id,
    )


def round_finalize_request_from_payload(
    payload: RoundFinalizeRequestPayload,
) -> RoundFinalizeRequest:
    """API payload를 domain finalize request로 변환한다."""
    return RoundFinalizeRequest(
        next_model_revision=payload.next_model_revision,
        next_auxiliary_artifact_versions=dict(payload.next_auxiliary_artifact_versions),
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

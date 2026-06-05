"""Shared adapter state publication capability."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from main_server.src.infrastructure.repositories import (
    shared_adapter_state_repository as state_repo_module,
)
from main_server.src.services.federation.rounds.active_manifest_service import (
    ActiveModelManifestService,
)
from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)
from main_server.src.services.federation.rounds.payload_adapters.models import (
    SharedAdapterRoundPayloadAdapter,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState

from .aggregation.artifact_payload_writer import save_aggregated_artifact_payload


@dataclass(frozen=True, slots=True)
class SharedAdapterStatePublicationRequest:
    """이미 계산된 next state projection을 server-owned state로 발행한다."""

    base_manifest: ModelManifest
    next_state: SharedAdapterState
    artifacts: Mapping[str, Mapping[str, Any]]
    published_at: datetime
    notes: str
    next_auxiliary_artifact_versions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SharedAdapterStatePublication:
    """server publication 이후 active manifest와 domain state."""

    next_manifest: ModelManifest
    next_state: SharedAdapterState


@dataclass(slots=True)
class SharedAdapterStatePublicationService:
    """server-owned artifact/state 저장과 active manifest 전환을 맡는다."""

    payload_adapter: SharedAdapterRoundPayloadAdapter
    state_repository: state_repo_module.SharedAdapterStateRepository
    active_manifest_service: ActiveModelManifestService
    artifact_store: AggregationArtifactStore

    def publish_projection(
        self,
        request: SharedAdapterStatePublicationRequest,
    ) -> SharedAdapterStatePublication:
        """projection artifact payload와 next state를 저장하고 active로 전환한다."""

        for artifact_ref, payload in request.artifacts.items():
            save_aggregated_artifact_payload(
                artifact_store=self.artifact_store,
                artifact_ref=artifact_ref,
                payload=dict(payload),
            )

        next_state_payload = self.payload_adapter.state_to_payload(request.next_state)
        self.state_repository.save_shared_adapter_state(next_state_payload)
        next_manifest = ModelManifest(
            schema_version=request.base_manifest.schema_version,
            model_id=request.base_manifest.model_id,
            model_revision=request.next_state.model_revision,
            published_at=request.published_at,
            artifact_kind="shared_adapter_state",
            artifact_ref=self.state_repository.ref_for_revision(
                request.next_state.model_revision
            ),
            auxiliary_artifact_versions=_build_next_auxiliary_artifact_versions(
                base_manifest=request.base_manifest,
                next_auxiliary_artifact_versions=(
                    request.next_auxiliary_artifact_versions
                ),
            ),
            training_scope=request.base_manifest.training_scope,
            training_enabled=request.base_manifest.training_enabled,
            compatible_task_types=request.base_manifest.compatible_task_types,
            base_model_id=request.base_manifest.base_model_id,
            base_model_revision=request.base_manifest.base_model_revision,
            notes=request.notes,
        )
        self.active_manifest_service.save_and_activate(
            next_manifest,
            activated_at=request.published_at,
        )
        return SharedAdapterStatePublication(
            next_manifest=next_manifest,
            next_state=request.next_state,
        )


def _build_next_auxiliary_artifact_versions(
    *,
    base_manifest: ModelManifest,
    next_auxiliary_artifact_versions: Mapping[str, str],
) -> dict[str, str]:
    result = dict(base_manifest.auxiliary_artifact_versions)
    result.update(
        {
            str(key): str(value)
            for key, value in next_auxiliary_artifact_versions.items()
            if str(key).strip() and str(value).strip()
        }
    )
    return result

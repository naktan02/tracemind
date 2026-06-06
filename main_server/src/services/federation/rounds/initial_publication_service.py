"""Initial shared adapter artifact publication service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from main_server.src.services.federation.rounds.active_manifest_service import (
    ActiveModelManifestService,
)
from main_server.src.services.federation.rounds.boundary.models import (
    InitialSharedArtifactPublicationRequest,
)
from main_server.src.services.federation.rounds.round_manager_service import (
    SharedAdapterStateRepository,
)
from methods.adaptation.initial_state import (
    SharedAdapterInitialStateRequest,
    build_initial_shared_adapter_state,
)
from shared.src.contracts.model_contracts import ArtifactKind, ModelManifest
from shared.src.domain.services.clock import Clock, SystemUtcClock


@dataclass(slots=True)
class InitialSharedArtifactPublicationService:
    """선택된 adapter family initial state를 active manifest로 publish한다."""

    artifact_repository: SharedAdapterStateRepository
    active_manifest_service: ActiveModelManifestService
    payload_adapter_kind: str
    update_family_name: str
    round_runtime_config: object | None = None
    clock: Clock = field(default_factory=SystemUtcClock)

    def publish(
        self,
        request: InitialSharedArtifactPublicationRequest,
    ) -> ModelManifest:
        """initial shared adapter state를 생성, 저장, active manifest로 publish한다."""

        published_at = self.clock.now()
        model_revision = request.model_revision or _new_initial_revision(published_at)
        state_payload = build_initial_shared_adapter_state(
            SharedAdapterInitialStateRequest(
                payload_adapter_kind=self.payload_adapter_kind,
                update_family_name=self.update_family_name,
                model_id=request.model_id,
                model_revision=model_revision,
                training_scope=str(request.training_scope),
                labels=request.label_schema,
                embedding_dim=request.embedding_dim,
                updated_at=published_at,
                round_runtime_config=self.round_runtime_config,
            )
        )
        self.artifact_repository.save_shared_adapter_state(state_payload)
        manifest = ModelManifest(
            model_id=request.model_id,
            model_revision=model_revision,
            published_at=published_at,
            artifact_kind=ArtifactKind.SHARED_ADAPTER_STATE,
            artifact_ref=self.artifact_repository.ref_for_revision(model_revision),
            training_scope=request.training_scope,
            training_enabled=True,
            compatible_task_types=request.compatible_task_types,
            notes=request.notes,
        )
        return self.active_manifest_service.save_and_activate(
            manifest,
            activated_at=published_at,
        )


def _new_initial_revision(published_at: datetime) -> str:
    timestamp = published_at.strftime("%Y%m%dT%H%M%SZ")
    return f"initial_{timestamp}_{uuid4().hex[:8]}"

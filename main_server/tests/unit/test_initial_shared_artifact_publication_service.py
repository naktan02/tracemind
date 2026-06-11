"""Initial shared artifact publication service tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

import main_server.src.api.fl_rounds as fl_rounds_api
from main_server.src.api.main import create_app
from main_server.src.infrastructure.repositories import (
    model_manifest_repository as model_manifest_repository_module,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_state_repository as shared_adapter_state_repository_module,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_update_repository as shared_adapter_update_repository_module,
)
from main_server.src.infrastructure.repositories.round_repository import RoundRepository
from main_server.src.services.federation.rounds.boundary.models import (
    InitialSharedArtifactPublicationRequest,
    RoundOpenDraftRequest,
)
from main_server.src.services.federation.rounds.boundary.payloads import (
    InitialSharedArtifactPublicationRequestPayload,
)
from main_server.src.services.federation.rounds.runtime.config import (
    ServerRoundRuntimeConfig,
)
from main_server.src.services.federation.rounds.runtime.factory import (
    build_round_lifecycle_service_from_config,
)
from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from shared.src.contracts.model_contracts import ArtifactKind
from shared.src.domain.services.clock import FixedClock

ModelManifestRepository = model_manifest_repository_module.ModelManifestRepository
SharedAdapterStateRepository = (
    shared_adapter_state_repository_module.SharedAdapterStateRepository
)
SharedAdapterUpdateRepository = (
    shared_adapter_update_repository_module.SharedAdapterUpdateRepository
)


def test_initial_publication_uses_methods_owned_peft_runtime_defaults(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 6, 7, 3, 0, tzinfo=timezone.utc)
    state_repository = SharedAdapterStateRepository(
        state_root=tmp_path / "shared_states"
    )
    service = build_round_lifecycle_service_from_config(
        ServerRoundRuntimeConfig(
            payload_adapter_kind="peft_classifier",
            update_family_name="peft_text_encoder",
            aggregation_backend_name="fedavg",
        ),
        round_repository=RoundRepository(state_root=tmp_path / "rounds"),
        update_payload_repository=SharedAdapterUpdateRepository(
            state_root=tmp_path / "updates"
        ),
        model_manifest_repository=ModelManifestRepository(
            state_root=tmp_path / "manifests"
        ),
        artifact_repository=state_repository,
        clock=FixedClock(fixed_time),
    )

    manifest = service.publish_initial_shared_artifact(
        InitialSharedArtifactPublicationRequest(
            model_id="tracemind-live",
            model_revision="rev_initial",
            label_schema=("anxiety", "normal"),
        )
    )

    assert manifest.artifact_kind == ArtifactKind.SHARED_ADAPTER_STATE
    assert manifest.artifact_ref == state_repository.ref_for_revision("rev_initial")
    state = state_repository.load_shared_adapter_state_from_ref(manifest.artifact_ref)
    peft_config = PeftEncoderTrainingBackendConfig()
    assert state.adapter_kind == "peft_classifier"
    assert state.model_revision == "rev_initial"
    assert state.label_schema == ["anxiety", "normal"]
    assert state.backbone.model_dump(mode="json") == (peft_config.to_backbone_payload())
    assert state.peft_adapter_config.model_dump(mode="json") == (
        peft_config.to_peft_adapter_config_payload()
    )

    opened = service.open_round(RoundOpenDraftRequest(round_id="round_initial"))
    assert opened.training_task.model_revision == "rev_initial"


def test_initial_publication_api_delegates_to_runtime_family_builder(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 6, 7, 3, 5, tzinfo=timezone.utc)
    state_repository = SharedAdapterStateRepository(
        state_root=tmp_path / "shared_states"
    )
    service = build_round_lifecycle_service_from_config(
        ServerRoundRuntimeConfig(
            payload_adapter_kind="peft_classifier",
            update_family_name="peft_text_encoder",
            aggregation_backend_name="fedavg",
        ),
        round_repository=RoundRepository(state_root=tmp_path / "rounds"),
        update_payload_repository=SharedAdapterUpdateRepository(
            state_root=tmp_path / "updates"
        ),
        model_manifest_repository=ModelManifestRepository(
            state_root=tmp_path / "manifests"
        ),
        artifact_repository=state_repository,
        clock=FixedClock(fixed_time),
    )

    response = fl_rounds_api.initialize_active_shared_artifact(
        InitialSharedArtifactPublicationRequestPayload(
            model_id="tracemind-live",
            model_revision="rev_api_initial",
            label_schema=["anxiety", "normal"],
        ),
        service=service,
    )

    assert response.model_revision == "rev_api_initial"
    state = state_repository.load_shared_adapter_state_from_ref(response.artifact_ref)
    assert state.backbone.backbone_model_id == (
        PeftEncoderTrainingBackendConfig().backbone_model_id
    )


def test_initial_publication_route_publishes_active_manifest(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 6, 7, 3, 10, tzinfo=timezone.utc)
    state_repository = SharedAdapterStateRepository(
        state_root=tmp_path / "shared_states"
    )
    service = build_round_lifecycle_service_from_config(
        ServerRoundRuntimeConfig(
            payload_adapter_kind="peft_classifier",
            update_family_name="peft_text_encoder",
            aggregation_backend_name="fedavg",
        ),
        round_repository=RoundRepository(state_root=tmp_path / "rounds"),
        update_payload_repository=SharedAdapterUpdateRepository(
            state_root=tmp_path / "updates"
        ),
        model_manifest_repository=ModelManifestRepository(
            state_root=tmp_path / "manifests"
        ),
        artifact_repository=state_repository,
        clock=FixedClock(fixed_time),
    )
    app = create_app(round_lifecycle_service=service)

    with TestClient(app, base_url="http://testserver") as client:
        response = client.post(
            "/api/v1/fl/rounds/active-manifest/initialize",
            json={
                "model_id": "tracemind-live",
                "model_revision": "rev_route_initial",
                "label_schema": ["anxiety", "normal"],
            },
        )

    assert response.status_code == 201
    assert response.json()["model_revision"] == "rev_route_initial"
    active = service.active_manifest_service.get_active_manifest()
    assert active.model_revision == "rev_route_initial"

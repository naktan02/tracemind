"""Agent runtime profile server API/service tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from main_server.src.api.main import create_app
from main_server.src.infrastructure.repositories.agent_runtime_profile_repository import (  # noqa: E501
    AgentRuntimeProfileRepository,
)
from main_server.src.services.agent_runtime_profile_service import (
    AgentRuntimeProfileService,
)
from shared.src.contracts.agent_runtime_profile_contracts import (
    AgentRuntimeProfilePayload,
    make_agent_runtime_profile_payload,
)
from shared.src.contracts.model_contracts import make_embedding_manifest
from shared.src.domain.services.clock import FixedClock


def _profile(
    *,
    profile_revision: str = "runtime_rev_001",
    model_revision: str = "clf_2026_04_11_143138",
    scorer_backend_name: str = "classifier_head_logits",
) -> AgentRuntimeProfilePayload:
    return make_agent_runtime_profile_payload(
        profile_id="profile_peft_classifier_lora",
        profile_revision=profile_revision,
        model_id="mixedbread-ai/mxbai-embed-large-v1",
        model_revision=model_revision,
        runtime_family="peft_classifier",
        adapter_mechanism="lora",
        scorer_backend_name=scorer_backend_name,
        embedding_backend="transformers_mxbai",
        embedding_model_id="mixedbread-ai/mxbai-embed-large-v1",
        training_scope="adapter_and_head",
        required_state_kind="peft_classifier_state.v2",
        updated_at=datetime(2026, 6, 13, tzinfo=timezone.utc),
    )


@dataclass(slots=True)
class _ActiveManifestService:
    def get_active_manifest(self):
        return make_embedding_manifest(
            model_id="mixedbread-ai/mxbai-embed-large-v1",
            model_revision="clf_2026_04_11_143138",
            artifact_ref="shared_adapter_state::clf_2026_04_11_143138",
            training_scope="adapter_and_head",
        )


def _service(tmp_path: Path) -> AgentRuntimeProfileService:
    return AgentRuntimeProfileService(
        repository=AgentRuntimeProfileRepository(
            state_root=tmp_path / "agent_runtime_profiles"
        ),
        active_manifest_service=_ActiveManifestService(),
        clock=FixedClock(datetime(2026, 6, 13, 9, 0, tzinfo=timezone.utc)),
    )


def test_runtime_profile_current_api_returns_saved_profile(tmp_path: Path) -> None:
    service = _service(tmp_path)
    profile = service.save_active_profile(_profile())
    app = create_app(agent_runtime_profile_service=service)

    with TestClient(app, base_url="http://testserver") as client:
        response = client.get("/api/v1/agent-runtime-profile/current")

    assert response.status_code == 200
    assert response.json()["profile_revision"] == profile.profile_revision
    assert response.json()["payload_checksum"] == profile.payload_checksum


def test_runtime_profile_validate_api_reports_stale_profile(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    current = service.save_active_profile(_profile(profile_revision="runtime_rev_002"))
    stale = _profile(profile_revision="runtime_rev_001")
    app = create_app(agent_runtime_profile_service=service)

    with TestClient(app, base_url="http://testserver") as client:
        response = client.post(
            "/api/v1/agent-runtime-profile/validate",
            json={
                "profile_id": stale.profile_id,
                "profile_revision": stale.profile_revision,
                "payload_checksum": stale.payload_checksum,
                "model_revision": stale.model_revision,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["up_to_date"] is False
    assert payload["latest_profile"]["profile_revision"] == current.profile_revision


def test_admin_runtime_profile_api_switches_active_profile(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    profile = _profile(scorer_backend_name="prototype_similarity")
    app = create_app(agent_runtime_profile_service=service)

    with TestClient(app, base_url="http://testserver") as client:
        response = client.post(
            "/api/v1/admin/runtime-profile",
            json=profile.model_dump(mode="json"),
        )
        current_response = client.get("/api/v1/admin/runtime-profile/current")

    assert response.status_code == 200
    assert response.json()["scorer_backend_name"] == "prototype_similarity"
    assert current_response.status_code == 200
    assert current_response.json()["payload_checksum"] == profile.payload_checksum


def test_runtime_profile_service_can_build_profile_from_env_and_manifest(
    tmp_path: Path,
) -> None:
    service = AgentRuntimeProfileService(
        repository=AgentRuntimeProfileRepository(
            state_root=tmp_path / "agent_runtime_profiles"
        ),
        active_manifest_service=_ActiveManifestService(),
        clock=FixedClock(datetime(2026, 6, 13, 9, 0, tzinfo=timezone.utc)),
        environ={
            "TRACEMIND_AGENT_RUNTIME_PROFILE_ID": "runtime_profile_env",
            "TRACEMIND_AGENT_RUNTIME_FAMILY": "peft_classifier",
            "TRACEMIND_AGENT_RUNTIME_ADAPTER_MECHANISM": "dora",
            "TRACEMIND_AGENT_RUNTIME_SCORER_BACKEND": "classifier_head_logits",
            "TRACEMIND_AGENT_RUNTIME_EMBEDDING_BACKEND": "transformers_mxbai",
            "TRACEMIND_AGENT_RUNTIME_REQUIRED_STATE_KIND": ("peft_classifier_state.v2"),
        },
    )

    profile = service.get_current_profile()

    assert profile.profile_id == "runtime_profile_env"
    assert profile.profile_revision == "clf_2026_04_11_143138"
    assert profile.adapter_mechanism == "dora"
    assert profile.training_scope == "adapter_and_head"

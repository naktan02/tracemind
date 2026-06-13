"""Agent runtime profile sync API/service tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.src.api.main import create_app
from agent.src.features.runtime_profile.repository import RuntimeProfileRepository
from agent.src.features.runtime_profile.sync_service import RuntimeProfileSyncService
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from agent.src.runtime.composition import build_agent_runtime_state
from shared.src.contracts.agent_runtime_profile_contracts import (
    AgentRuntimeProfilePayload,
    make_agent_runtime_profile_payload,
)


def _profile(
    *,
    profile_revision: str = "runtime_rev_001",
) -> AgentRuntimeProfilePayload:
    return make_agent_runtime_profile_payload(
        profile_id="profile_peft_classifier_lora",
        profile_revision=profile_revision,
        model_id="mixedbread-ai/mxbai-embed-large-v1",
        model_revision="clf_2026_04_11_143138",
        runtime_family="peft_classifier",
        adapter_mechanism="lora",
        scorer_backend_name="peft_classifier_head_logits",
        embedding_backend="transformers_mxbai",
        embedding_model_id="mixedbread-ai/mxbai-embed-large-v1",
        training_scope="adapter_and_head",
        required_state_kind="peft_classifier_state.v2",
        updated_at=datetime(2026, 6, 13, tzinfo=timezone.utc),
    )


def test_runtime_profile_sync_fetches_profile_and_pulls_shared_state(
    tmp_path: Path,
) -> None:
    repository = RuntimeProfileRepository(db_path=tmp_path / "agent_local.db")
    shared_adapter_sync_service = MagicMock()
    profile = _profile()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/agent-runtime-profile/current"
        return httpx.Response(
            status_code=200,
            request=request,
            content=profile.model_dump_json().encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    service = RuntimeProfileSyncService(
        repository=repository,
        shared_adapter_sync_service=shared_adapter_sync_service,
        _transport=httpx.MockTransport(handler),
    )

    result = service.sync_current(server_base_url="http://server.test")

    assert result.status == "updated"
    assert result.profile is not None
    assert result.profile.identity_matches(profile)
    shared_adapter_sync_service.pull_current.assert_called_once_with(
        server_base_url="http://server.test"
    )
    active = repository.load_active()
    assert active is not None
    assert active.profile.identity_matches(profile)
    assert active.server_validated_at is not None
    assert active.server_base_url == "http://server.test"


def test_runtime_profile_sync_refreshes_shared_state_when_up_to_date(
    tmp_path: Path,
) -> None:
    repository = RuntimeProfileRepository(db_path=tmp_path / "agent_local.db")
    profile = _profile()
    repository.save_profile(profile, source="server", activate=True)
    shared_adapter_sync_service = MagicMock()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/agent-runtime-profile/validate"
        request_payload = json.loads(request.content.decode("utf-8"))
        assert request_payload["profile_revision"] == profile.profile_revision
        return httpx.Response(
            status_code=200,
            request=request,
            json={"up_to_date": True, "latest_profile": None},
        )

    service = RuntimeProfileSyncService(
        repository=repository,
        shared_adapter_sync_service=shared_adapter_sync_service,
        _transport=httpx.MockTransport(handler),
    )

    result = service.sync_current(server_base_url="http://server.test")

    assert result.status == "up_to_date"
    shared_adapter_sync_service.pull_current.assert_called_once_with(
        server_base_url="http://server.test"
    )
    active = repository.load_active()
    assert active is not None
    assert active.server_validated_at is not None
    assert active.server_base_url == "http://server.test"


def test_runtime_profile_sync_api_updates_agent_local_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = RuntimeProfileRepository(db_path=tmp_path / "agent_local.db")
    profile = _profile()
    shared_adapter_sync_service = MagicMock()
    pipeline_service = MagicMock()
    monkeypatch.setattr(
        "agent.src.api.runtime_profile.build_pipeline_service_from_runtime_profile",
        lambda **_kwargs: pipeline_service,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            request=request,
            content=profile.model_dump_json().encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    sync_service = RuntimeProfileSyncService(
        repository=repository,
        shared_adapter_sync_service=shared_adapter_sync_service,
        _transport=httpx.MockTransport(handler),
    )
    app = create_app(
        runtime_profile_repository=repository,
        runtime_profile_sync_service=sync_service,
        auto_configure_pipeline=False,
    )

    with TestClient(app, base_url="http://testserver") as client:
        initial = client.get("/api/v1/runtime-profile/status")
        response = client.post(
            "/api/v1/runtime-profile/sync",
            json={"server_base_url": "http://server.test"},
        )

    assert initial.status_code == 200
    assert initial.json()["has_active_profile"] is False
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "updated"
    assert payload["active_profile"]["has_active_profile"] is True
    assert (
        payload["active_profile"]["profile"]["profile_revision"]
        == profile.profile_revision
    )
    assert app.state.pipeline_service is pipeline_service


def test_startup_auto_configure_validates_profile_and_rebuilds_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = RuntimeProfileRepository(db_path=tmp_path / "agent_local.db")
    repository.save_profile(
        _profile(),
        source="server",
        activate=True,
        server_base_url="http://server.test",
    )
    sync_service = MagicMock()
    pipeline_service = MagicMock()
    captured_text_view_generation_service = MagicMock()
    captured_text_view_generation_service.translation_provider = None
    observed_server_base_urls = []

    def _build_pipeline(**kwargs):
        observed_server_base_urls.append(kwargs["server_base_url"])
        return pipeline_service

    monkeypatch.setattr(
        "agent.src.runtime.composition.build_pipeline_service_from_runtime_profile",
        _build_pipeline,
    )

    runtime_state = build_agent_runtime_state(
        analysis_event_repository=AnalysisEventRepository(
            db_path=tmp_path / "events.db"
        ),
        runtime_profile_repository=repository,
        runtime_profile_sync_service=sync_service,
        shared_adapter_runtime_service=MagicMock(),
        captured_text_view_generation_service=captured_text_view_generation_service,
        auto_configure_pipeline=True,
    )

    sync_service.sync_current.assert_called_once_with(
        server_base_url="http://server.test"
    )
    assert observed_server_base_urls == ["http://server.test"]
    assert runtime_state.pipeline_service is pipeline_service

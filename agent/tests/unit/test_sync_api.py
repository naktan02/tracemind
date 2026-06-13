"""Sync API tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import agent.src.api.sync as sync_api
from agent.src.api.main import app
from agent.src.infrastructure.repositories.shared_adapter_state_repository import (
    SharedAdapterStateRepository,
)
from agent.src.services.assets.shared_adapters.runtime_service import (
    SharedAdapterRuntimeService,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_peft_classifier_state_payload,
)
from shared.src.contracts.model_contracts import make_embedding_manifest


def _peft_state(*, model_revision: str):
    return make_peft_classifier_state_payload(
        model_id="model",
        model_revision=model_revision,
        backbone={
            "backbone_model_id": "mixedbread-ai/mxbai-embed-large-v1",
            "backbone_revision": "main",
            "tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
            "tokenizer_revision": "main",
            "pooling": "mean",
            "max_length": 256,
            "task_prefix": "",
        },
        peft_adapter_config={
            "peft_adapter_name": "lora",
            "parameters": {"rank": 8},
        },
        label_schema=["anxiety", "normal"],
    )


def test_sync_api_reads_current_local_shared_adapter_state(tmp_path: Path) -> None:
    repository = SharedAdapterStateRepository(state_root=tmp_path / "shared_states")
    state = _peft_state(model_revision="rev_001")
    repository.save_current(
        manifest=make_embedding_manifest(
            model_id="model",
            model_revision="rev_001",
            auxiliary_artifact_versions={},
            artifact_ref="/server/state/rev_001.json",
            training_scope=state.training_scope,
        ),
        state=state,
    )

    response = sync_api.get_current_local_shared_adapter_state(
        runtime_service=SharedAdapterRuntimeService(repository=repository)
    )

    assert response.manifest.model_revision == "rev_001"
    assert response.state.model_revision == "rev_001"


def test_sync_api_maps_remote_errors_to_http_exceptions() -> None:
    sync_service = MagicMock()
    sync_service.pull_current.side_effect = RuntimeError("connection reset")

    with pytest.raises(HTTPException) as error_info:
        sync_api.pull_current_shared_adapter_state(
            sync_api.AssetPullRequest(server_base_url="http://testserver"),
            sync_service=sync_service,
        )

    assert error_info.value.status_code == 502


def test_sync_router_is_registered_on_agent_app() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/sync/prototypes/current" not in route_paths
    assert "/api/v1/sync/prototypes/pull" not in route_paths
    assert "/api/v1/sync/shared-adapters/current" in route_paths
    assert "/api/v1/sync/shared-adapters/pull" in route_paths

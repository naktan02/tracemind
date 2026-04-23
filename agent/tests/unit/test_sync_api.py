"""Sync API tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import agent.src.api.sync as sync_api
from agent.src.api.main import app
from agent.src.infrastructure.repositories.prototype_pack_repository import (
    PrototypePackRepository,
)
from agent.src.services.prototypes.runtime_service import PrototypeRuntimeService
from shared.src.contracts.prototype_contracts import PrototypePackPayload


def _build_payload() -> PrototypePackPayload:
    return PrototypePackPayload.model_validate(
        {
            "schema_version": "prototype_pack.v1",
            "prototype_version": "proto_test_v1",
            "embedding_model_id": "hash_debug",
            "embedding_model_revision": "main",
            "mapping_version": "ourafla_to_4cat.v1",
            "build_method": "mean_centroid_l2_normalized",
            "distance_metric": "cosine",
            "built_at": datetime(2026, 4, 2, tzinfo=timezone.utc),
            "categories": {
                "anxiety": [
                    {
                        "prototype_id": "anxiety:single",
                        "centroid": [1.0, 0.0],
                        "sample_count": 2,
                    }
                ]
            },
        }
    )


def test_sync_api_reads_current_local_pack(tmp_path: Path) -> None:
    repository = PrototypePackRepository(state_root=tmp_path / "prototype_packs")
    payload = _build_payload()
    repository.save_pack(payload)
    repository.set_active(payload.prototype_version)

    response = sync_api.get_current_local_prototype_pack(
        runtime_service=PrototypeRuntimeService(repository=repository)
    )
    assert response.prototype_version == payload.prototype_version


def test_sync_api_maps_remote_errors_to_http_exceptions() -> None:
    sync_service = MagicMock()
    sync_service.pull_current.side_effect = RuntimeError("connection reset")

    with pytest.raises(HTTPException) as error_info:
        sync_api.pull_current_prototype_pack(
            sync_api.PrototypePullRequest(server_base_url="http://testserver"),
            sync_service=sync_service,
        )

    assert error_info.value.status_code == 502


def test_sync_router_is_registered_on_agent_app() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/sync/prototypes/current" in route_paths
    assert "/api/v1/sync/prototypes/pull" in route_paths

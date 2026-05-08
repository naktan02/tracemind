"""Prototype API tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

import main_server.src.api.prototypes as prototypes_api
from main_server.src.api.main import app
from main_server.src.infrastructure.repositories.prototype_pack_repository import (
    PrototypePackRepository,
)
from main_server.src.services.federation.prototypes.prototype_pack_service import (
    PrototypePackService,
)
from shared.src.contracts.prototype_contracts import (
    PrototypePackActivationRequest,
    PrototypePackPayload,
)


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


def _build_service(tmp_path: Path) -> tuple[PrototypePackService, PrototypePackPayload]:
    repository = PrototypePackRepository(state_root=tmp_path / "prototype_packs")
    payload = _build_payload()
    repository.save_pack(payload)
    service = PrototypePackService(repository=repository)
    return service, payload


def test_prototypes_api_get_and_activate_flow(tmp_path: Path) -> None:
    service, payload = _build_service(tmp_path)

    loaded = prototypes_api.get_prototype_pack(
        payload.prototype_version,
        service=service,
    )
    assert loaded.prototype_version == payload.prototype_version

    pointer = prototypes_api.activate_prototype_pack(
        PrototypePackActivationRequest(prototype_version=payload.prototype_version),
        service=service,
    )
    assert pointer.prototype_version == payload.prototype_version

    current = prototypes_api.get_current_prototype_pack(service=service)
    assert current.pack.prototype_version == payload.prototype_version


def test_prototypes_api_returns_404_without_active_pack(tmp_path: Path) -> None:
    repository = PrototypePackRepository(state_root=tmp_path / "prototype_packs")
    service = PrototypePackService(repository=repository)

    with pytest.raises(HTTPException) as error_info:
        prototypes_api.get_current_prototype_pack(service=service)
    assert error_info.value.status_code == 404


def test_prototypes_router_is_registered_on_main_app() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/prototypes/current" in route_paths
    assert "/api/v1/prototypes/{prototype_version}" in route_paths
    assert "/api/v1/prototypes/activate" in route_paths

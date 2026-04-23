"""Prototype runtime/sync unit tests."""

from __future__ import annotations

from pathlib import Path

import httpx

from agent.src.infrastructure.repositories.prototype_pack_repository import (
    PrototypePackRepository,
)
from agent.src.services.prototypes.runtime_service import PrototypeRuntimeService
from agent.src.services.prototypes.sync_service import PrototypeSyncService
from shared.src.contracts.prototype_contracts import (
    CurrentPrototypePackResponse,
    PrototypePackActivationPointer,
    PrototypePackPayload,
)


def _load_fixture_payload() -> PrototypePackPayload:
    fixture_path = Path(__file__).with_name("fixtures") / "prototype_pack_v1.json"
    return PrototypePackPayload.model_validate_json(
        fixture_path.read_text(encoding="utf-8")
    )


def _transport_with_json(
    payload: dict[str, object] | None,
    *,
    status_code: int = 200,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if payload is None:
            return httpx.Response(status_code=status_code, request=request)
        return httpx.Response(
            status_code=status_code,
            request=request,
            json=payload,
        )

    return httpx.MockTransport(handler)


def _multi_payload() -> PrototypePackPayload:
    return PrototypePackPayload.model_validate(
        {
            "schema_version": "prototype_pack.v1",
            "prototype_version": "proto_multi_v1",
            "embedding_model_id": "hash_debug",
            "embedding_model_revision": "main",
            "mapping_version": "ourafla_to_4cat.v1",
            "build_method": "kmeans_mean_centroid_l2_normalized",
            "distance_metric": "cosine",
            "built_at": "2026-04-02T00:00:00+00:00",
            "categories": {
                "anxiety": [
                    {
                        "prototype_id": "anxiety:kmeans:0",
                        "centroid": [1.0, 0.0, 0.0],
                        "sample_count": 2,
                    },
                    {
                        "prototype_id": "anxiety:kmeans:1",
                        "centroid": [0.0, 1.0, 0.0],
                        "sample_count": 5,
                    },
                ],
                "normal": [
                    {
                        "prototype_id": "normal:single",
                        "centroid": [0.0, 0.0, 1.0],
                        "sample_count": 3,
                    }
                ],
            },
        }
    )


def test_runtime_service_reads_active_pack(tmp_path: Path) -> None:
    repository = PrototypePackRepository(state_root=tmp_path / "prototype_packs")
    payload = _load_fixture_payload()
    repository.save_pack(payload)
    repository.set_active(payload.prototype_version)

    runtime_service = PrototypeRuntimeService(repository=repository)

    centroids = runtime_service.get_active_single_centroids()

    assert sorted(centroids) == ["anxiety", "depression", "normal", "suicidal"]
    assert centroids["anxiety"] == [1.0, 0.0, 0.0]


def test_runtime_service_reads_active_prototype_lists(tmp_path: Path) -> None:
    repository = PrototypePackRepository(state_root=tmp_path / "prototype_packs")
    payload = _load_fixture_payload()
    repository.save_pack(payload)
    repository.set_active(payload.prototype_version)

    runtime_service = PrototypeRuntimeService(repository=repository)

    prototypes = runtime_service.get_active_prototypes()

    assert sorted(prototypes) == ["anxiety", "depression", "normal", "suicidal"]
    assert prototypes["anxiety"] == ([1.0, 0.0, 0.0],)


def test_runtime_service_projects_representative_centroids_for_multi_pack(
    tmp_path: Path,
) -> None:
    repository = PrototypePackRepository(state_root=tmp_path / "prototype_packs")
    payload = _multi_payload()
    repository.save_pack(payload)
    repository.set_active(payload.prototype_version)

    runtime_service = PrototypeRuntimeService(repository=repository)

    projected = runtime_service.get_active_projected_centroids()

    assert projected["anxiety"] == [0.0, 1.0, 0.0]
    assert projected["normal"] == [0.0, 0.0, 1.0]


def test_sync_service_pulls_current_pack_and_activates_it(tmp_path: Path) -> None:
    repository = PrototypePackRepository(state_root=tmp_path / "prototype_packs")
    payload = _load_fixture_payload()
    response_payload = CurrentPrototypePackResponse(
        active=PrototypePackActivationPointer(
            prototype_version=payload.prototype_version,
            activated_at=payload.built_at,
        ),
        pack=payload,
    )
    sync_service = PrototypeSyncService(
        repository=repository,
        _transport=_transport_with_json(response_payload.model_dump(mode="json")),
    )
    pointer = sync_service.pull_current(server_base_url="http://127.0.0.1:8000/")

    assert pointer.prototype_version == payload.prototype_version
    assert repository.path_for_version(payload.prototype_version).exists()

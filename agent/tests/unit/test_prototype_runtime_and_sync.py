"""Prototype runtime/sync unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from agent.src.infrastructure.repositories.prototype_pack_repository import (
    PrototypePackRepository,
)
from agent.src.services.prototype.runtime_service import PrototypeRuntimeService
from agent.src.services.prototype.sync_service import PrototypeSyncService
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


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_runtime_service_reads_active_pack(tmp_path: Path) -> None:
    repository = PrototypePackRepository(state_root=tmp_path / "prototype_packs")
    payload = _load_fixture_payload()
    repository.save_pack(payload)
    repository.set_active(payload.prototype_version)

    runtime_service = PrototypeRuntimeService(repository=repository)

    centroids = runtime_service.get_active_single_centroids()

    assert sorted(centroids) == ["anxiety", "depression", "normal", "suicidal"]
    assert centroids["anxiety"] == [1.0, 0.0, 0.0]


def test_runtime_service_keeps_single_centroid_alias(tmp_path: Path) -> None:
    repository = PrototypePackRepository(state_root=tmp_path / "prototype_packs")
    payload = _load_fixture_payload()
    repository.save_pack(payload)
    repository.set_active(payload.prototype_version)

    runtime_service = PrototypeRuntimeService(repository=repository)

    assert runtime_service.get_active_centroids() == (
        runtime_service.get_active_single_centroids()
    )


def test_runtime_service_reads_active_prototype_lists(tmp_path: Path) -> None:
    repository = PrototypePackRepository(state_root=tmp_path / "prototype_packs")
    payload = _load_fixture_payload()
    repository.save_pack(payload)
    repository.set_active(payload.prototype_version)

    runtime_service = PrototypeRuntimeService(repository=repository)

    prototypes = runtime_service.get_active_prototypes()

    assert sorted(prototypes) == ["anxiety", "depression", "normal", "suicidal"]
    assert prototypes["anxiety"] == ([1.0, 0.0, 0.0],)


def test_sync_service_pulls_current_pack_and_activates_it(tmp_path: Path) -> None:
    repository = PrototypePackRepository(state_root=tmp_path / "prototype_packs")
    sync_service = PrototypeSyncService(repository=repository)
    payload = _load_fixture_payload()
    response_payload = CurrentPrototypePackResponse(
        active=PrototypePackActivationPointer(
            prototype_version=payload.prototype_version,
            activated_at=payload.built_at,
        ),
        pack=payload,
    )

    with patch(
        "agent.src.services.prototype.sync_service.urlopen",
        return_value=_FakeResponse(response_payload.model_dump(mode="json")),
    ):
        pointer = sync_service.pull_current(server_base_url="http://127.0.0.1:8000/")

    assert pointer.prototype_version == payload.prototype_version
    assert repository.path_for_version(payload.prototype_version).exists()
